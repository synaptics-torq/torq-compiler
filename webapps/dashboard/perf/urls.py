from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .api import TestCaseViewSet, TestSessionViewSet, TestRunViewSet

router = DefaultRouter()
router.register(r'test-cases', TestCaseViewSet)
router.register(r'test-sessions', TestSessionViewSet)
router.register(r'test-runs', TestRunViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
