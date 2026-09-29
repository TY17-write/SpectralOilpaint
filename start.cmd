@echo off
cd /d "%~dp0"
echo Open http://localhost:8080 in Chrome or Edge.
echo Press Ctrl+C in this window to stop the server.
node server.mjs
pause
