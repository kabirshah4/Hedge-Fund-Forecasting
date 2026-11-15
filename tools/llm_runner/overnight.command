#!/bin/bash
# Double-click this file at bedtime. It runs the LLM jobs until 7:00 AM, then stops.
# Keep the Mac plugged in with the lid open (the screen can turn off).
cd "$(dirname "$0")"
open -a Ollama 2>/dev/null; sleep 5
caffeinate -i python3 run_jobs.py --until 07:00
echo; echo "You can close this window."
