@echo off
chcp 866

rem this batch is used to call exe, with PATH added

rem ==================== exe name, same as batch name ====================
set BATCH_PATH=%~dp0
set EXE_NAME=%1
shift

set ADDITIONAL_PATH=%BATCH_PATH%..\3rdParty\bin\win64;

set EXE_PATH=%BATCH_PATH%..\bin\Release\%EXE_NAME%.exe

rem ==================== check whether EXE exist ====================
if not exist "%EXE_PATH%" (
	echo "%EXE_PATH%" not exist
	goto end_of_bat
)

set PATH=%ADDITIONAL_PATH%%PATH%

rem ==================== compact all arguments ====================
set ARGUMENTS=%1
shift
:extract_argument_loop
if "%~1"=="" goto after_extract_argument_loop
set ARGUMENTS=%ARGUMENTS% %1
shift
goto extract_argument_loop
:after_extract_argument_loop


rem ==================== run the exe ====================
echo %EXE_PATH% %ARGUMENTS%

%EXE_PATH% %ARGUMENTS%


:end_of_bat