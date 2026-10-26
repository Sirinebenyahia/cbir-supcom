@echo off
set PROJET=C:\Users\sirin\cbir
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" cbir
cd /d "%PROJET%"
start "Serveur CBIR (FastAPI)" /D "%PROJET%" cmd /k uvicorn api:app --app-dir backend --port 8000
timeout /t 8 /nobreak >nul
start "Interface CBIR (Streamlit)" /D "%PROJET%" cmd /k streamlit run frontend\app.py