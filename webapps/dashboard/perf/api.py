import os
import zipfile
import json
from tempfile import TemporaryDirectory
from django.core.files.base import File
from rest_framework.parsers import MultiPartParser
import requests
from rest_framework import viewsets, filters, status
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination
from rest_framework.decorators import action
from .models import TestCase, TestSession, TestRun
from .serializers import TestCaseSerializer, TestSessionSerializer, TestRunSerializer


class StandardResultsSetPagination(PageNumberPagination):
    page_size = 100
    page_size_query_param = 'page_size'
    max_page_size = 1000


class TestCaseViewSet(viewsets.ModelViewSet):
    queryset = TestCase.objects.all()
    serializer_class = TestCaseSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['module', 'name']
    ordering_fields = ['module', 'name']
    ordering = ['module', 'name']


class TestSessionViewSet(viewsets.ModelViewSet):
    queryset = TestSession.objects.all()
    serializer_class = TestSessionSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['git_commit', 'git_branch']
    ordering_fields = ['timestamp', 'id', 'git_branch']
    ordering = ['-timestamp']

    @action(detail=False, methods=['post'], parser_classes=[MultiPartParser])
    def upload_zip(self, request):
        if 'file' not in request.FILES:
            return Response({'detail': 'No file provided.'}, status=status.HTTP_400_BAD_REQUEST)

        upload = request.FILES['file']
        if not upload.name.endswith('.zip'):
            return Response({'detail': 'File is not a zip archive.'}, status=status.HTTP_400_BAD_REQUEST)

        with TemporaryDirectory() as temp_dir:

            with zipfile.ZipFile(upload, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Find JSON manifest named test_session.json
            manifest_path = os.path.join(temp_dir, 'test_session.json')
            
            if not os.path.exists(manifest_path):
                return Response({'detail': 'Required manifest test_session.json not found in archive.'}, status=status.HTTP_400_BAD_REQUEST)

            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            git_commit = manifest.get('git_commit')
            git_branch = manifest.get('git_branch')
            workflow_url = manifest.get('workflow_url')
            owner = manifest.get('owner')

            # Reuse get_or_create logic for consistency with unique workflow_url
            if workflow_url:
                test_session, _ = TestSession.objects.get_or_create(
                    workflow_url=workflow_url,
                    defaults={
                        'owner': owner,
                        'git_commit': git_commit,
                        'git_branch': git_branch,
                    },
                )
            else:
                test_session = TestSession.objects.create(
                    owner=owner,
                    git_commit=git_commit,
                    git_branch=git_branch,
                    workflow_url=None,
                )

            outcome_map = {
                'passed': TestRun.Outcome.PASS,
                'failed': TestRun.Outcome.FAIL,
                'skipped': TestRun.Outcome.SKIP,
            }

            test_runs = manifest.get('test_runs', [])
            
            for item in test_runs:                
                module = item.get('module', '')
                name = item.get('name', '')
                parameters = item.get('parameters', '')
                outcome = item.get('outcome', '')
                profiling_rel = item.get('profiling_file')  # relative path inside zip (optional)

                if not module or not name or outcome not in outcome_map:
                    raise Response({'detail': 'Invalid run entry in manifest.'}, status=status.HTTP_400_BAD_REQUEST)
                
                test_case, _ = TestCase.objects.get_or_create(
                    module=str(module), name=str(name), parameters=str(parameters)
                )

                test_run = TestRun.objects.create(
                    test_session=test_session,
                    test_case=test_case,
                    outcome=outcome_map[outcome],
                )

                if profiling_rel:
                    profiling_path = os.path.join(os.path.dirname(manifest_path), profiling_rel)

                    if not os.path.exists(profiling_path):
                        raise Response({'detail': f'Profiling file {profiling_rel} not found in archive.'}, status=status.HTTP_400_BAD_REQUEST)
                
                    with open(profiling_path, 'rb') as pf:
                        test_run.profiling_data.save(os.path.basename(profiling_path), File(pf), save=True)

            payload = self.get_serializer(test_session).data

            return Response(payload, status=status.HTTP_201_CREATED)
        

class TestRunViewSet(viewsets.ModelViewSet):
    queryset = TestRun.objects.all()
    serializer_class = TestRunSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['test_case__name', 'test_case__module']
    ordering_fields = ['test_session', 'test_case', 'outcome']
    ordering = ['-test_session__timestamp']

