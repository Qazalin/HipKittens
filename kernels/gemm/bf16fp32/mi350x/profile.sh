PYTHONPATH=. rocprofv3 \
  --kernel-trace \
  --stats \
  --output-format csv \
  --output-directory /tmp/rocprof \
  -- python test_min.py
