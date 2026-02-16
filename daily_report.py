"""
일일 매매 결과 리포트 - 이메일로 전송.

실행: python daily_report.py
스케줄: crontab 등으로 매일 09:00 (한국시간)에 실행하세요.

.env 설정 필요:
  EMAIL_TO=받을이메일@example.com
  SMTP_HOST=smtp.gmail.com (또는 smtp.naver.com)
  SMTP_PORT=587
  SMTP_USER=보내는이메일
  SMTP_PASSWORD=비밀번호
"""

import csv
import json
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOG_FILE = Path("trades_log.csv")
KST = timezone(timedelta(hours=9))
REGIME_KR = {"sideways": "횡보장", "trend": "추세장", "neutral": "중립"}


def _parse_meta(meta_str: str) -> dict:
    """meta 컬럼 JSON 문자열을 dict로 파싱"""
    if not meta_str or not meta_str.strip():
        return {}
    try:
        return json.loads(meta_str)
    except (json.JSONDecodeError, ValueError):
        # 이전 형식(repr) 호환성 유지
        try:
            import ast
            return ast.literal_eval(meta_str)
        except (ValueError, SyntaxError):
            return {}


def _to_kst(dt_utc: datetime) -> datetime:
    """UTC datetime을 KST로 변환"""
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(KST)


def get_previous_day_trades() -> list[dict]:
    """전날(한국시간 기준) 거래 내역 조회. CSV가 시간 순이면 전날 구간 지나면 읽기 중단해서 대량 기록 시에도 빠르게 동작."""
    if not LOG_FILE.exists():
        return []

    today_kst = datetime.now(KST).date()
    yesterday = today_kst - timedelta(days=1)
    rows = []
    with LOG_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts_str = row["time_utc"]
                # ISO 형식 파싱 (Z 또는 +00:00 처리)
                if ts_str.endswith("Z"):
                    ts_str = ts_str[:-1] + "+00:00"
                elif "+" not in ts_str and "-" in ts_str[-6:]:
                    ts_str = ts_str + "+00:00"
                ts = datetime.fromisoformat(ts_str)
            except (ValueError, KeyError) as e:
                continue
            ts_kst = _to_kst(ts)
            # 시간 순으로 정렬되어 있다고 가정: 전날 이후 날짜가 나오면 중단
            if ts_kst.date() > yesterday:
                break  # 시간 순이므로 이후 행은 모두 오늘 이후 → 조기 종료
            if ts_kst.date() == yesterday:
                row["time_kst"] = ts_kst
                row["meta_dict"] = _parse_meta(row.get("meta", "{}"))
                rows.append(row)
    return rows


def summarize_trades(trades: list[dict]) -> dict:
    """거래 내역 요약"""
    if not trades:
        return {"total_trades": 0, "total_pnl": 0, "wins": 0, "losses": 0}

    total_pnl = sum(float(t.get("pnl", 0) or 0) for t in trades)
    wins = sum(1 for t in trades if float(t.get("pnl", 0) or 0) > 0)
    losses = sum(1 for t in trades if float(t.get("pnl", 0) or 0) < 0)

    # 마지막 거래 후 잔고
    last_balance = float(trades[-1].get("balance_after", 0) or 0) if trades else 0

    return {
        "total_trades": len(trades),
        "total_pnl": total_pnl,
        "wins": wins,
        "losses": losses,
        "last_balance": last_balance,
        "win_rate": (wins / len(trades) * 100) if trades else 0,
    }


def build_report_html(trades: list[dict], summary: dict) -> str:
    """이메일 본문 HTML 생성 (거래별 상세: 수익률, 잔고, 장세, 청산사유 포함)"""
    yesterday = (datetime.now(KST).date() - timedelta(days=1)).strftime("%Y-%m-%d")

    if summary["total_trades"] == 0:
        return f"""
        <h2>매매 결과 요약 ({yesterday})</h2>
        <p>전날 거래 내역이 없습니다.</p>
        <p style="color:#666;font-size:12px">Binance Futures Bot - 일일 리포트</p>
        """

    rows_html = ""
    for t in trades:
        pnl = float(t.get("pnl", 0) or 0)
        pnl_class = "green" if pnl > 0 else "red"
        meta = t.get("meta_dict") or {}
        pnl_pct = meta.get("pnl_pct")
        pnl_pct_str = f"{pnl_pct:+.2f}%" if pnl_pct is not None else "-"
        regime = meta.get("regime", "")
        regime_kr = REGIME_KR.get(regime, regime or "-")
        reason = meta.get("reason", "")
        balance_after = float(t.get("balance_after", 0) or 0)
        entry_p = float(t.get("entry_price", 0) or 0)
        exit_p = float(t.get("exit_price", 0) or 0)
        rows_html += f"""
        <tr>
            <td>{t['time_kst'].strftime('%H:%M')}</td>
            <td>{t.get('side', '-')}</td>
            <td>{entry_p:,.2f}</td>
            <td>{exit_p:,.2f}</td>
            <td style="color:{pnl_class}">{pnl_pct_str}</td>
            <td style="color:{pnl_class}">{pnl:+.2f}</td>
            <td>{balance_after:,.2f}</td>
            <td>{regime_kr}</td>
            <td style="font-size:11px">{reason[:40] + '…' if reason and len(reason) > 40 else (reason or '-')}</td>
        </tr>
        """

    return f"""
    <h2>매매 결과 요약 ({yesterday})</h2>
    <p><strong>총 거래:</strong> {summary['total_trades']}회 | 
       <strong>승:</strong> {summary['wins']} / <strong>패:</strong> {summary['losses']} | 
       <strong>승률:</strong> {summary['win_rate']:.1f}%</p>
    <p><strong>총 손익:</strong> {summary['total_pnl']:+.2f} USDT | 
       <strong>최종 잔고:</strong> {summary['last_balance']:,.2f} USDT</p>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;font-size:13px">
        <thead>
            <tr style="background:#eee">
                <th>시간</th><th>방향</th><th>진입가</th><th>청산가</th><th>수익률</th><th>손익</th><th>잔고(청산후)</th><th>장세</th><th>청산사유</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    <p style="color:#666;font-size:12px">Binance Futures Bot - 일일 리포트</p>
    """


def send_email(subject: str, html_body: str) -> bool:
    """이메일 발송"""
    import os

    to_addr = os.getenv("EMAIL_TO", "")
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASSWORD")

    if not to_addr or not smtp_user or not smtp_pass:
        print("[ERROR] .env 에 EMAIL_TO, SMTP_USER, SMTP_PASSWORD 설정이 필요합니다.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_addr
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_addr, msg.as_string())
        print(f"[OK] 이메일 전송 완료: {to_addr}")
        return True
    except Exception as e:
        print(f"[ERROR] 이메일 전송 실패: {e}")
        return False


def run_report() -> bool:
    """전날 매매 결과 리포트 생성 및 이메일 전송"""
    trades = get_previous_day_trades()
    summary = summarize_trades(trades)
    yesterday = (datetime.now(KST).date() - timedelta(days=1)).strftime("%Y-%m-%d")

    subject = f"[Binance Bot] 매매 결과 {yesterday} - {summary['total_trades']}건, {summary['total_pnl']:+.2f} USDT"
    html = build_report_html(trades, summary)
    return send_email(subject, html)


if __name__ == "__main__":
    run_report()
