import html
import time

from django.db import connection


class QueryDebugFooterMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start_time = time.perf_counter()
        start_query_count = len(connection.queries)
        response = self.get_response(request)

        if self._should_skip(request, response):
            return response

        query_entries = connection.queries[start_query_count:]
        total_query_time_ms = 0.0
        lines = []

        for index, entry in enumerate(query_entries, start=1):
            query_time_s = float(entry.get('time', 0.0) or 0.0)
            query_time_ms = query_time_s * 1000.0
            total_query_time_ms += query_time_ms   
            
            raw_sql = entry.get('sql', '') or ''
            sql = html.escape(raw_sql)
            
            lines.append(
                '<div style="padding: 8px 0; border-bottom: 1px solid #1f2937;">'
                f'<div style="font-size: 12px; color: #93c5fd; margin-bottom: 4px;">Query {index} - {query_time_ms:.3f} ms</div>'
                f'<pre style="margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 12px; color: #e5e7eb;">{sql}</pre>'
                '</div>'
            )

        if not lines:
            lines.append('<div style="font-size: 12px; color: #9ca3af;">No SQL queries were recorded for this page.</div>')

        page_time_ms = (time.perf_counter() - start_time) * 1000.0
        footer_html = (
            '<details style="position: fixed; left: 12px; right: 12px; bottom: 12px; z-index: 9999; background: #111827; color: #e5e7eb; border: 1px solid #374151; border-radius: 10px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);">'
            '<summary style="cursor: pointer; padding: 12px 14px; font-weight: 600; outline: none;">'
            f'Query Debug: {len(query_entries)} queries, {total_query_time_ms:.3f} ms SQL, {page_time_ms:.3f} ms total'
            '</summary>'
            '<div style="max-height: 45vh; overflow: auto; border-top: 1px solid #374151; padding: 10px 14px 12px 14px;">'
            + ''.join(lines) +
            '</div>'
            '</details>'
        )

        charset = response.charset or 'utf-8'
        content = response.content.decode(charset, errors='replace')
        lower_content = content.lower()
        body_end = lower_content.rfind('</body>')
        if body_end >= 0:
            content = content[:body_end] + footer_html + content[body_end:]
        else:
            content += footer_html

        response.content = content.encode(charset)
        if 'Content-Length' in response:
            response['Content-Length'] = str(len(response.content))
        return response

    def _should_skip(self, request, response):
        if getattr(response, 'streaming', False):
            return True

        content_type = response.get('Content-Type', '')
        if 'text/html' not in content_type.lower():
            return True

        resolver_match = getattr(request, 'resolver_match', None)
        if resolver_match is None:
            return True

        view_func = getattr(resolver_match, 'func', None)
        view_module = getattr(view_func, '__module__', '')
        return not view_module.startswith('perf.views')
