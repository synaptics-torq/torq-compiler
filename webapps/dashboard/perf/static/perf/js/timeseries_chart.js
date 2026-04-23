function formatTimeseriesMeasurement(value, unit) {
    if (value === null || value === undefined || value === '') {
        return 'N/A';
    }

    if (unit === 'ns') {
        const absoluteValue = Math.abs(value);
        if (absoluteValue >= 1_000_000) {
            return `${(value / 1_000_000).toFixed(2)} ms`;
        }
        if (absoluteValue >= 1_000) {
            return `${(value / 1_000).toFixed(2)} us`;
        }
        if (Number.isInteger(value)) {
            return `${value} ns`;
        }
        return `${value.toFixed(2)} ns`;
    }

    if (unit === '%') {
        return `${value.toFixed(2)} %`;
    }

    if (unit === 'unitless' || !unit) {
        if (Number.isInteger(value)) {
            return String(value);
        }
        return value.toFixed(2);
    }

    return `${value.toFixed(2)} ${unit}`;
}

function formatTimeseriesDate(timestamp, options = {}) {
    const { includeYear = false, includeTime = false } = options;
    const parts = {
        month: 'short',
        day: 'numeric',
    };

    if (includeYear) {
        parts.year = 'numeric';
    }

    if (includeTime) {
        parts.hour = '2-digit';
        parts.minute = '2-digit';
    }

    return new Intl.DateTimeFormat(undefined, parts).format(new Date(timestamp));
}

function timeseriesPointColor(point) {
    if (point.is_current) {
        return '#0d6efd';
    }
    if (point.is_baseline) {
        return '#f59f00';
    }
    return '#198754';
}

function getTimeseriesPointValue(point, seriesName = null) {
    if (seriesName && point.values) {
        const value = point.values[seriesName];
        return value === undefined ? null : value;
    }

    if (point.value !== undefined) {
        return point.value;
    }

    if (!seriesName && point.values) {
        const firstValue = Object.values(point.values)[0];
        return firstValue === undefined ? null : firstValue;
    }

    return null;
}

function getTimeseriesPointDetails(point) {
    const details = [];

    if (point.session_id) {
        details.push(`Session #${point.session_id}`);
    }
    if (point.test_run_id) {
        details.push(`Run #${point.test_run_id}`);
    }
    if (point.outcome) {
        details.push(`Outcome: ${point.outcome}`);
    }
    if (point.git_branch) {
        details.push(`Branch: ${point.git_branch}`);
    }
    if (point.git_commit) {
        details.push(`Commit: ${point.git_commit}`);
    }
    if (point.is_current) {
        details.push('Current comparison point');
    } else if (point.is_baseline) {
        details.push('Baseline comparison point');
    }

    return details;
}

function syncTimeseriesFormFields(form, root) {
    if (!form) {
        return;
    }

    const metricInput = form.querySelector('#id_metric');
    const startDateInput = form.querySelector('#id_start_date');
    const endDateInput = form.querySelector('#id_end_date');
    const metricSelect = root.querySelector('.js-timeseries-metric-select');
    const startSessionSelect = root.querySelector('.js-timeseries-start-session');
    const endSessionSelect = root.querySelector('.js-timeseries-end-session');

    if (metricInput && metricSelect) {
        metricInput.value = metricSelect.value;
    }

    if (startDateInput && startSessionSelect?.selectedOptions?.[0]) {
        startDateInput.value = startSessionSelect.selectedOptions[0].dataset.timestamp || '';
    }

    if (endDateInput && endSessionSelect?.selectedOptions?.[0]) {
        endDateInput.value = endSessionSelect.selectedOptions[0].dataset.timestamp || '';
    }
}

function submitTimeseriesForm(form) {
    if (!form) {
        return;
    }

    if (typeof form.requestSubmit === 'function') {
        form.requestSubmit();
        return;
    }

    form.submit();
}

function normalizeTimeseriesSessionWindow(startSessionSelect, endSessionSelect, changedSelect) {
    if (!startSessionSelect || !endSessionSelect) {
        return;
    }

    if (startSessionSelect.selectedIndex <= endSessionSelect.selectedIndex) {
        return;
    }

    if (changedSelect === startSessionSelect) {
        endSessionSelect.selectedIndex = startSessionSelect.selectedIndex;
        return;
    }

    startSessionSelect.selectedIndex = endSessionSelect.selectedIndex;
}

function bindSingleMetricTimeseriesControls(root) {
    const form = root.querySelector('.js-timeseries-form');
    if (!form) {
        return;
    }

    const rangeButtons = Array.from(root.querySelectorAll('[data-history-range]'));
    const metricSelect = root.querySelector('.js-timeseries-metric-select');
    const startSessionSelect = root.querySelector('.js-timeseries-start-session');
    const endSessionSelect = root.querySelector('.js-timeseries-end-session');

    rangeButtons.forEach(button => {
        button.addEventListener('click', () => {
            if (!startSessionSelect || !endSessionSelect || startSessionSelect.options.length === 0) {
                return;
            }

            const rangeKey = button.dataset.historyRange;
            const sessionOptions = Array.from(startSessionSelect.options).map(option => ({
                sessionId: option.value,
                timestamp: option.dataset.timestamp,
            }));
            const latestSession = sessionOptions[sessionOptions.length - 1];
            let startSession = sessionOptions[0];

            if (rangeKey !== 'all') {
                const days = Number.parseInt(rangeKey, 10);
                const cutoffTime = new Date(latestSession.timestamp).getTime() - (days * 24 * 60 * 60 * 1000);
                startSession = sessionOptions.find(option => new Date(option.timestamp).getTime() >= cutoffTime) || startSession;
            }

            startSessionSelect.value = String(startSession.sessionId);
            endSessionSelect.value = String(latestSession.sessionId);
            syncTimeseriesFormFields(form, root);
            submitTimeseriesForm(form);
        });
    });

    if (metricSelect) {
        metricSelect.addEventListener('change', () => {
            syncTimeseriesFormFields(form, root);
            submitTimeseriesForm(form);
        });
    }

    if (startSessionSelect && endSessionSelect) {
        startSessionSelect.addEventListener('change', () => {
            normalizeTimeseriesSessionWindow(startSessionSelect, endSessionSelect, startSessionSelect);
            syncTimeseriesFormFields(form, root);
            submitTimeseriesForm(form);
        });

        endSessionSelect.addEventListener('change', () => {
            normalizeTimeseriesSessionWindow(startSessionSelect, endSessionSelect, endSessionSelect);
            syncTimeseriesFormFields(form, root);
            submitTimeseriesForm(form);
        });
    }
}

function updateSingleMetricSummary(root, timeseries, selectedMetric) {
    const rangeLabelElement = root.querySelector('.js-timeseries-range-label');
    const rangeChangeElement = root.querySelector('.js-timeseries-range-change');
    const rangeMetaElement = root.querySelector('.js-timeseries-range-meta');
    if (!rangeChangeElement || !rangeMetaElement) {
        return;
    }

    const metricPoints = timeseries.points.filter(point => {
        const value = getTimeseriesPointValue(point, selectedMetric.name);
        return value !== null && value !== undefined;
    });

    if (rangeLabelElement && timeseries.range_label) {
        rangeLabelElement.textContent = timeseries.range_label;
    }

    if (metricPoints.length === 0) {
        rangeChangeElement.className = 'h5 mb-0 text-body-secondary';
        rangeChangeElement.textContent = 'No data in selected range';
        rangeMetaElement.textContent = `${selectedMetric.label} is unavailable for the selected sessions in this range.`;
        return;
    }

    const firstPoint = metricPoints[0];
    const lastPoint = metricPoints[metricPoints.length - 1];
    const lastValue = getTimeseriesPointValue(lastPoint, selectedMetric.name);

    if (metricPoints.length < 2) {
        rangeChangeElement.className = 'h5 mb-0 text-body-secondary';
        rangeChangeElement.textContent = `${selectedMetric.label}: ${formatTimeseriesMeasurement(lastValue, selectedMetric.unit)}`;
        rangeMetaElement.textContent = 'Only one session falls inside the selected server-side history window for the selected metric.';
        return;
    }

    const firstValue = getTimeseriesPointValue(firstPoint, selectedMetric.name);
    const deltaValue = lastValue - firstValue;
    const deltaPercent = firstValue === 0
        ? null
        : (deltaValue * 100.0) / firstValue;
    const directionLabel = deltaValue < 0
        ? 'Improved'
        : deltaValue > 0
            ? 'Regressed'
            : 'Unchanged';
    const directionClass = deltaValue < 0
        ? 'text-success'
        : deltaValue > 0
            ? 'text-danger'
            : 'text-body-secondary';
    const deltaSuffix = deltaPercent === null
        ? ''
        : ` (${deltaPercent >= 0 ? '+' : ''}${deltaPercent.toFixed(2)}%)`;

    rangeChangeElement.className = `h5 mb-0 ${directionClass}`;
    rangeChangeElement.textContent = `${selectedMetric.label}: ${directionLabel} ${deltaValue >= 0 ? '+' : ''}${formatTimeseriesMeasurement(deltaValue, selectedMetric.unit)}${deltaSuffix}`;
    rangeMetaElement.textContent = `${metricPoints.length} sessions from ${formatTimeseriesDate(firstPoint.timestamp, { includeYear: true })} to ${formatTimeseriesDate(lastPoint.timestamp, { includeYear: true })}.`;
}

function renderSingleMetricTimeseries(root, timeseries, canvas) {
    const selectedMetric = timeseries.selected_metric;
    if (!selectedMetric || !timeseries.points.length) {
        return;
    }

    updateSingleMetricSummary(root, timeseries, selectedMetric);

    new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
            labels: timeseries.points.map(point => formatTimeseriesDate(point.timestamp)),
            datasets: [
                {
                    label: selectedMetric.label,
                    data: timeseries.points.map(point => getTimeseriesPointValue(point, selectedMetric.name)),
                    borderColor: '#0b7285',
                    backgroundColor: 'rgba(11, 114, 133, 0.12)',
                    fill: true,
                    tension: 0.25,
                    pointRadius: timeseries.points.map(point => point.is_current ? 6 : point.is_baseline ? 5 : 3),
                    pointHoverRadius: timeseries.points.map(point => point.is_current ? 8 : point.is_baseline ? 7 : 5),
                    pointBackgroundColor: timeseries.points.map(timeseriesPointColor),
                    pointBorderColor: timeseries.points.map(timeseriesPointColor),
                    pointHitRadius: 14,
                },
            ],
        },
        options: {
            maintainAspectRatio: false,
            interaction: {
                mode: 'nearest',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    callbacks: {
                        title(tooltipItems) {
                            const point = timeseries.points[tooltipItems[0].dataIndex];
                            return formatTimeseriesDate(point.timestamp, { includeYear: true });
                        },
                        label(tooltipItem) {
                            const point = timeseries.points[tooltipItem.dataIndex];
                            return `${selectedMetric.label}: ${formatTimeseriesMeasurement(getTimeseriesPointValue(point, selectedMetric.name), selectedMetric.unit)}`;
                        },
                        afterLabel(tooltipItem) {
                            const point = timeseries.points[tooltipItem.dataIndex];
                            return getTimeseriesPointDetails(point);
                        },
                    },
                },
            },
            scales: {
                y: {
                    ticks: {
                        callback(value) {
                            return formatTimeseriesMeasurement(value, selectedMetric.unit);
                        },
                    },
                },
                x: {
                    ticks: {
                        maxRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 8,
                    },
                },
            },
        },
    });
}

function renderMultiSeriesTimeseries(timeseries, canvas) {
    if (!timeseries.points.length || !timeseries.datasets?.length) {
        return;
    }

    const allUnitless = timeseries.datasets.every(dataset => dataset.unit === 'unitless');

    new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
            labels: timeseries.points.map(point => formatTimeseriesDate(point.timestamp, { includeYear: true, includeTime: true })),
            datasets: timeseries.datasets.map(dataset => ({
                label: dataset.label,
                data: timeseries.points.map(point => getTimeseriesPointValue(point, dataset.name)),
                borderColor: dataset.borderColor,
                backgroundColor: dataset.backgroundColor,
                tension: 0.2,
                fill: false,
                pointRadius: 3,
                pointHoverRadius: 5,
                borderWidth: 2,
            })),
        },
        options: {
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        afterBody(tooltipItems) {
                            const point = timeseries.points[tooltipItems[0].dataIndex];
                            return getTimeseriesPointDetails(point);
                        },
                        label(tooltipItem) {
                            const dataset = timeseries.datasets[tooltipItem.datasetIndex];
                            return `${dataset.label}: ${formatTimeseriesMeasurement(tooltipItem.parsed.y, dataset.unit)}`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    title: {
                        display: Boolean(timeseries.x_axis_label),
                        text: timeseries.x_axis_label,
                    },
                },
                y: {
                    beginAtZero: allUnitless,
                    title: {
                        display: Boolean(timeseries.y_axis_label),
                        text: timeseries.y_axis_label,
                    },
                    ticks: allUnitless
                        ? { precision: 0 }
                        : {
                            callback(value) {
                                return formatTimeseriesMeasurement(value, timeseries.datasets[0].unit);
                            },
                        },
                },
            },
        },
    });
}

function initializeTimeseriesRoot(root) {
    const jsonScriptId = root.dataset.timeseriesJsonId;
    const historyDataElement = jsonScriptId ? document.getElementById(jsonScriptId) : null;
    const historyCanvas = root.querySelector('.js-timeseries-canvas');

    if (!historyDataElement || !historyCanvas) {
        return;
    }

    const timeseries = JSON.parse(historyDataElement.textContent);

    if (timeseries.kind === 'single_metric') {
        bindSingleMetricTimeseriesControls(root);
        renderSingleMetricTimeseries(root, timeseries, historyCanvas);
        return;
    }

    renderMultiSeriesTimeseries(timeseries, historyCanvas);
}

document.addEventListener('DOMContentLoaded', () => {
    if (typeof Chart === 'undefined') {
        return;
    }

    document.querySelectorAll('[data-timeseries-root]').forEach(initializeTimeseriesRoot);
});
