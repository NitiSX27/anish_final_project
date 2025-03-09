# Gunicorn configuration file
bind = "0.0.0.0:10000"
workers = 1
threads = 2
timeout = 120
preload_app = False
worker_class = "sync"
loglevel = "info"
