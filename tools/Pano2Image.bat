@echo off

rem ========================================
rem 	this batch calls another batch, by adding
rem	the batch name (without extension) as second argument
rem ========================================

set BATCH_PATH=%~dp0
set EXE_NAME=%~n0

rem ==================== run the command ====================
echo %BATCH_PATH%EXE.bat %EXE_NAME% %*

%BATCH_PATH%EXE.bat %EXE_NAME% %*