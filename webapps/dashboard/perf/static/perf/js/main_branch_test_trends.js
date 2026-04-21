document.addEventListener('DOMContentLoaded', function() {
    for (const canvas of document.querySelectorAll('canvas.main-branch-test-trends-chart')) {
        renderMainBranchTestTrends(canvas);
    }
});

function renderMainBranchTestTrends(canvas) {
    const ctx = canvas.getContext('2d');
    const chartData = JSON.parse(canvas.dataset.chart);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels.map(value => {
                const date = new Date(value);
                return date.toLocaleString([], {
                    year: 'numeric',
                    month: 'short',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                });
            }),
            datasets: chartData.datasets.map(dataset => ({
                ...dataset,
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
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Session timestamp',
                    },
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of tests',
                    },
                    ticks: {
                        precision: 0,
                    },
                },
            },
            plugins: {
                legend: {
                    position: 'top',
                },
            },
        },
    });
}