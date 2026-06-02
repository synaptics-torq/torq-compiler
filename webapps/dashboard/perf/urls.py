from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .api import TestCaseViewSet, TestSessionViewSet, TestRunViewSet, TestGroupViewSet
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

from . import views

router = DefaultRouter()
router.register(r'test-cases', TestCaseViewSet)
router.register(r'test-sessions', TestSessionViewSet)
router.register(r'test-runs', TestRunViewSet)
router.register(r'test-groups', TestGroupViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/schema/swagger-ui/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('health/', views.health, name='health'),
    path('main-branch-test-trends/', views.main_branch_test_trends, name='main_branch_test_trends'),
    path('test-sessions/<int:session_id>/', views.test_session, name='test_session'),
    path('test-sessions/<int:session_id>/summary', views.test_session_summary, name='test_session_summary'),
    path('test-sessions/<int:session_id>/metric_details', views.test_session_metric_details, name='test_session_metric_details'),    
    path('test-sessions/<int:session_id>/results', views.test_session_results, name='test_session_results'),    
    path('test-runs/<int:test_run_id>/', views.test_run, name='test_run'),
    path('download-trace/<int:test_run_id>/', views.download_trace, name='download_trace'),
    path('download-failure-log/<int:test_run_id>/', views.download_failure_log, name='download_failure_log'),
    path('', views.home, name='home'),
]
