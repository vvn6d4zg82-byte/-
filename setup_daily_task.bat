@echo off
chcp 65001 >nul
echo ===== 每日21:30自动生成研究日志 =====
echo.
echo 正在创建任务计划...
echo.

schtasks /create /tn "生成研究日志" /tr "python \"C:\Users\周正\Desktop\33550336\Ayxi\Ayin\eyes\auto_daily_log.py\"" /sc daily /st 21:30 /f

echo.
echo 已创建！每天21:30会自动运行生成日志
echo.
pause