"""
uWSGI spoolers of the dashboard, started by uwsgi.ini with one process each:

- RESULTS_SPOOLER imports the uploaded test results. It is the first one in uwsgi.ini, so the default
  of .spool().
- MAINTENANCE_SPOOLER runs the backups, restores and trims, so that they don't delay the results.
"""

# must match the spooler lines of uwsgi.ini
RESULTS_SPOOLER = b'/data/spool'
MAINTENANCE_SPOOLER = b'/data/spool-maintenance'
