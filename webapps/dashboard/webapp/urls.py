"""
URL configuration for webapp project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

import perf.backup
import perf.trim
import perf.urls

urlpatterns = [
    path('admin/database/backup/', admin.site.admin_view(perf.backup.backup_database_view), name='admin_database_backup'),
    path('admin/database/backup/upload/', admin.site.admin_view(perf.backup.upload_backup_view), name='admin_database_backup_upload'),
    path('admin/database/backup/<str:backup_id>/download/', admin.site.admin_view(perf.backup.download_backup_view), name='admin_database_backup_download'),
    path('admin/database/backup/<str:backup_id>/restore/', admin.site.admin_view(perf.backup.restore_backup_view), name='admin_database_backup_restore'),
    path('admin/database/backup/<str:backup_id>/delete/', admin.site.admin_view(perf.backup.delete_backup_view), name='admin_database_backup_delete'),
    path('admin/database/trim/', admin.site.admin_view(perf.trim.trim_database_view), name='admin_database_trim'),
    path('admin/', admin.site.urls),
] + perf.urls.urlpatterns
