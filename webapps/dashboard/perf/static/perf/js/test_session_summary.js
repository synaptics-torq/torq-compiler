document.addEventListener('DOMContentLoaded', function() {

    for (const canvas of document.querySelectorAll('canvas.change-histogram-chart')) {            
        renderChangeHistograms(canvas);
    }

});

function renderChangeHistograms(canvas) {

    const ctx = canvas.getContext('2d');

    const histogramData = JSON.parse(canvas.dataset.chart);
    
    const data = {
        labels: histogramData.map(bucket => bucket.label),
        datasets: [{
            label: 'Count of tests',
            data: histogramData.map(bucket => bucket.count),
            backgroundColor: histogramData.map(bucket => bucket.lower_bound !== null && bucket.lower_bound >= 0 ? 'rgb(220, 53, 69)' : 'rgb(25, 135, 84)'),
            borderWidth: 1
        }]
    };

    const config = {
        type: 'bar',
        data: data,
        options: {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Change (%)'
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Count of tests'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }                
            }
        },
    };

    new Chart(ctx, config);
}

document.addEventListener('DOMContentLoaded', function() {
    renderDurationChangeHistogram();
});