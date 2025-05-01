#!/bin/bash
gunicorn --timeout 600 --workers 4 --bind 0.0.0.0:$PORT app:app