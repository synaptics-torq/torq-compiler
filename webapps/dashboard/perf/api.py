import os
import zipfile
import json
from tempfile import TemporaryDirectory, NamedTemporaryFile
from django.core.files.base import File
from rest_framework.parsers import MultiPartParser
import requests
from rest_framework import viewsets, filters, status
from rest_framework.response import Response
from rest_framework.pagination import PageNumberPagination
from rest_framework.decorators import action
from django.db.models import Count, Q
from .models import TestCase, TestSession, TestRun, TestRunBatch
from .serializers import TestCaseSerializer, TestSessionSerializer, TestRunSerializer
from .processing import process_uploaded_zip


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
    ordering_fields = ['module', 'name', 'parameters', 'id']
    ordering = ['module', 'name', 'parameters', 'id']


class TestSessionViewSet(viewsets.ModelViewSet):
    queryset = TestSession.objects.annotate(
        num_total=Count('testrunbatch__testrun'),
        num_passed=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.PASS)),
        num_failed=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.FAIL)),
        num_skipped=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.SKIP)),
        num_error=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.ERROR)),
        num_xfail=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.XFAIL)),
        num_nxpass=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.NXPASS))
    ).all()
    serializer_class = TestSessionSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['git_commit', 'git_branch']
    ordering_fields = ['timestamp', 'id', 'git_branch']
    ordering = ['-timestamp']

    @action(detail=False, methods=['post'], parser_classes=[MultiPartParser])
    def mark_complete(self, request):
        """
        Marks a test session as complete, indicating that all batches have been uploaded and processed.

        Expects a multipart/form-data body with the following field:
            workflow_url: https://ci.example.com/workflow/12345

        The workflow_url is used to identify the test session to mark as complete.
        """
        workflow_url = request.data.get('workflow_url')
        if not workflow_url:
            return Response({'detail': 'workflow_url is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            test_session = TestSession.objects.get(workflow_url=workflow_url)
            test_session.completed = True
            test_session.save()
            return Response({'detail': f'Test session with workflow_url {workflow_url} marked as complete.'}, status=status.HTTP_200_OK)
        except TestSession.DoesNotExist:
            return Response({'detail': f'No test session found with workflow_url {workflow_url}.'}, status=status.HTTP_404_NOT_FOUND)


    @action(detail=False, methods=['post'], parser_classes=[MultiPartParser])
    def upload_zip(self, request):
        if 'file' not in request.FILES:
            return Response({'detail': 'No file provided.'}, status=status.HTTP_400_BAD_REQUEST)

        upload = request.FILES['file']
        if not upload.name.endswith('.zip'):
            return Response({'detail': 'File is not a zip archive.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            with zipfile.ZipFile(upload, 'r') as zip_ref:
                if 'test_session.json' not in zip_ref.namelist():
                    return Response({'detail': 'Required manifest test_session.json not found in archive.'}, status=status.HTTP_400_BAD_REQUEST)
                
                with zip_ref.open('test_session.json') as f:
                    manifest = json.load(f)
        except zipfile.BadZipFile:
            return Response({'detail': 'Invalid zip file.'}, status=status.HTTP_400_BAD_REQUEST)
        except json.JSONDecodeError:
            return Response({'detail': 'Invalid JSON in manifest.'}, status=status.HTTP_400_BAD_REQUEST)

        # Extract metadata from manifest or request
        git_commit = manifest.get('git_commit')
        git_branch = manifest.get('git_branch')
        workflow_url = manifest.get('workflow_url')
        test_plan = manifest.get('test_plan', 'torq')
        owner = manifest.get('owner')

        # Reuse get_or_create logic for consistency with unique workflow_url
        if workflow_url:
            test_session, _ = TestSession.objects.get_or_create(
                workflow_url=workflow_url,
                defaults={
                    'owner': owner,
                    'git_commit': git_commit,
                    'git_branch': git_branch,
                    'test_plan': test_plan
                },
            )
        else:
            test_session = TestSession.objects.create(
                owner=owner,
                git_commit=git_commit,
                git_branch=git_branch,
                workflow_url=None,
                test_plan=test_plan,
                completed=True # Mark as complete immediately since these tests have a single batch
            )

        # Create a TestRunBatch for this upload
        test_run_batch = TestRunBatch.objects.create(
            test_session=test_session,
            name=manifest.get('batch_name'),
            processed=False,
        )

        # Save the uploaded file to a temporary location
        # The spooler will process the test runs
        with NamedTemporaryFile(delete=False, suffix='.zip', dir='/tmp') as temp_file:
            upload.seek(0)  # Reset file pointer
            for chunk in upload.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name

        # Spool the test run processing job
        process_uploaded_zip.spool(
            zip_path=temp_path.encode('utf-8'),
            test_run_batch_id=str(test_run_batch.id).encode('utf-8'),
        )

        # Return the serialized test session
        payload = self.get_serializer(test_session).data
        return Response(payload, status=status.HTTP_201_CREATED)
        

class TestRunViewSet(viewsets.ModelViewSet):
    queryset = TestRun.objects.all()
    serializer_class = TestRunSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['test_case__name', 'test_case__module']
    ordering_fields = ['test_run_batch__test_session', 'test_case', 'outcome']
    ordering = ['-test_run_batch__test_session__timestamp']

