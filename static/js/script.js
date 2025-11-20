// static/js/script.js - ENHANCED BITCOIN PREDICTOR VERSION - FIXED FEATURE CATEGORIES

// Global variables for charts
let priceChart, sentimentChart, performanceChart, featureChart, confidenceChart;
let predictionHistory = [];

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    injectEnhancedStyles(); // Inject enhanced styles first
    initializeCharts();
    checkStatus();
    loadDashboardData();
    updateDataFreshness();
    forceChartResize(); // Add chart resize
});

// Tab switching function
function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(tabName).classList.add('active');
    
    // Add active class to clicked tab
    event.currentTarget.classList.add('active');
    
    // Load data for specific tabs
    if (tabName === 'analytics') {
        console.log('🔍 Switching to Analytics tab');
        setTimeout(() => {
            loadAnalyticsData();
            // Force chart resize for proper rendering
            setTimeout(forceChartResize, 200);
        }, 100);
    } else if (tabName === 'history') {
        loadHistoryData();
    }
    
    // Force chart resize after tab switch
    setTimeout(forceChartResize, 100);
}

// Force chart resize and proper rendering
function forceChartResize() {
    setTimeout(() => {
        if (priceChart) {
            priceChart.resize();
            priceChart.update('none');
        }
        if (sentimentChart) {
            sentimentChart.resize();
            sentimentChart.update('none');
        }
        if (performanceChart) {
            performanceChart.resize();
            performanceChart.update('none');
        }
        if (featureChart) {
            featureChart.resize();
            featureChart.update('none');
        }
        if (confidenceChart) {
            confidenceChart.resize();
            confidenceChart.update('none');
        }
    }, 500);
}

// Initialize all charts with enhanced visualization
function initializeCharts() {
    // ENHANCED Price History Chart with Time Series - FIXED MARGINS
    const priceCtx = document.getElementById('priceChart').getContext('2d');
    priceChart = new Chart(priceCtx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Bitcoin Price (USD)',
                data: [],
                borderColor: '#F7931A',
                backgroundColor: 'rgba(247, 147, 26, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.2,
                pointBackgroundColor: '#F7931A',
                pointBorderColor: '#FFFFFF',
                pointBorderWidth: 2,
                pointRadius: 3,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            layout: {
                padding: {
                    top: 10,
                    right: 10,
                    bottom: 25, // Increased bottom padding for x-axis
                    left: 10
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        font: {
                            size: 12,
                            family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
                        },
                        color: '#333'
                    }
                },
                title: {
                    display: true,
                    text: 'Bitcoin Price History',
                    color: '#F7931A',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleFont: {
                        size: 12
                    },
                    bodyFont: {
                        size: 12
                    },
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += new Intl.NumberFormat('en-US', {
                                    style: 'currency',
                                    currency: 'USD',
                                    minimumFractionDigits: 0,
                                    maximumFractionDigits: 0
                                }).format(context.parsed.y);
                            }
                            return label;
                        },
                        title: function(tooltipItems) {
                            const date = new Date(tooltipItems[0].parsed.x);
                            return date.toLocaleDateString('en-US', {
                                weekday: 'short',
                                year: 'numeric',
                                month: 'short',
                                day: 'numeric'
                            });
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: {
                            day: 'MMM dd'
                        },
                        tooltipFormat: 'MMM dd, yyyy'
                    },
                    grid: {
                        color: 'rgba(0,0,0,0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#6c757d',
                        font: {
                            size: 11
                        },
                        maxRotation: 45,
                        minRotation: 45,
                        padding: 10 // Added padding for x-axis labels
                    },
                    title: {
                        display: true,
                        text: 'Date',
                        color: '#6c757d',
                        font: {
                            size: 12,
                            weight: 'bold'
                        },
                        padding: {
                            top: 10,
                            bottom: 5
                        }
                    }
                },
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(0,0,0,0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#6c757d',
                        font: {
                            size: 11
                        },
                        padding: 10, // Added padding for y-axis labels
                        callback: function(value) {
                            if (value >= 1000000) {
                                return '$' + (value / 1000000).toFixed(1) + 'M';
                            } else if (value >= 1000) {
                                return '$' + (value / 1000).toFixed(0) + 'K';
                            }
                            return '$' + value;
                        }
                    },
                    title: {
                        display: true,
                        text: 'Price (USD)',
                        color: '#6c757d',
                        font: {
                            size: 12,
                            weight: 'bold'
                        },
                        padding: {
                            top: 5,
                            bottom: 10
                        }
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            elements: {
                line: {
                    tension: 0.2
                }
            }
        }
    });
    
    // ENHANCED Sentiment Chart - FIXED MARGINS
    const sentimentCtx = document.getElementById('sentimentChart').getContext('2d');
    sentimentChart = new Chart(sentimentCtx, {
        type: 'bar',
        data: {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [{
                label: 'Wikipedia Sentiment (Last 30 Days)',
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.8)',
                    'rgba(108, 117, 125, 0.8)',
                    'rgba(247, 147, 26, 0.8)'
                ],
                borderColor: [
                    'rgb(40, 167, 69)',
                    'rgb(108, 117, 125)',
                    'rgb(247, 147, 26)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            layout: {
                padding: {
                    top: 10,
                    right: 10,
                    bottom: 25, // Increased bottom padding
                    left: 10
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Wikipedia Sentiment Analysis',
                    color: '#F7931A',
                    font: {
                        size: 14,
                        weight: 'bold'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed.y / total) * 100).toFixed(1);
                            return `${context.dataset.label}: ${context.parsed.y} days (${percentage}%)`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Days',
                        color: '#6c757d',
                        font: {
                            size: 12,
                            weight: 'bold'
                        },
                        padding: {
                            top: 5,
                            bottom: 10
                        }
                    },
                    ticks: {
                        color: '#6c757d',
                        padding: 10
                    },
                    grid: {
                        color: 'rgba(0,0,0,0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: '#6c757d',
                        padding: 10
                    },
                    grid: {
                        color: 'rgba(0,0,0,0.1)'
                    }
                }
            }
        }
    });
    
    // ENHANCED Performance Chart
   // ENHANCED Performance Chart
    const performanceCtx = document.getElementById('performanceChart').getContext('2d');
    performanceChart = new Chart(performanceCtx, {
        type: 'doughnut',
        data: {
            labels: ['Correct Predictions', 'Incorrect Predictions'],
            datasets: [{
                data: [0, 0],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.8)',
                    'rgba(247, 147, 26, 0.8)'
                ],
                borderColor: [
                    'rgb(40, 167, 69)',
                    'rgb(247, 147, 26)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Model Performance',
                    color: '#F7931A',
                    font: {
                        size: 14,
                        weight: 'bold'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed;
                            // Handle both percentage and count data
                            if (value <= 100) {
                                // Percentage data
                                return `${context.label}: ${value.toFixed(1)}%`;
                            } else {
                                // Count data
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((context.parsed / total) * 100).toFixed(1);
                                return `${context.label}: ${context.parsed} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        }
    });
    
    // ENHANCED Feature Importance Chart
    const featureCtx = document.getElementById('featureChart').getContext('2d');
    featureChart = new Chart(featureCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Feature Importance',
                data: [],
                backgroundColor: 'rgba(247, 147, 26, 0.8)',
                borderColor: 'rgb(247, 147, 26)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Feature Importance',
                    color: '#F7931A',
                    font: {
                        size: 14,
                        weight: 'bold'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Importance: ${(context.parsed.x * 100).toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Importance Score',
                        color: '#6c757d',
                        font: {
                            size: 12,
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        color: '#6c757d',
                        callback: function(value) {
                            return (value * 100).toFixed(1) + '%';
                        }
                    },
                    grid: {
                        color: 'rgba(0,0,0,0.1)'
                    }
                },
                y: {
                    ticks: {
                        color: '#6c757d'
                    },
                    grid: {
                        color: 'rgba(0,0,0,0.1)'
                    }
                }
            }
        }
    });
    
    // ENHANCED Confidence Distribution Chart
    const confidenceCtx = document.getElementById('confidenceChart').getContext('2d');
    confidenceChart = new Chart(confidenceCtx, {
        type: 'polarArea',
        data: {
            labels: ['High (70-100%)', 'Medium (50-70%)', 'Low (0-50%)'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.8)',
                    'rgba(247, 147, 26, 0.8)',
                    'rgba(220, 53, 69, 0.8)'
                ],
                borderColor: [
                    'rgb(40, 167, 69)',
                    'rgb(247, 147, 26)',
                    'rgb(220, 53, 69)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Confidence Distribution',
                    color: '#F7931A',
                    font: {
                        size: 14,
                        weight: 'bold'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = total > 0 ? ((context.parsed.r / total) * 100).toFixed(1) : 0;
                            return `${context.label}: ${context.parsed.r} predictions (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Load all dashboard data
async function loadDashboardData() {
    await loadPriceHistory();
    await loadSentimentData();
    await loadPerformanceData();
    updateDataFreshness();
}

// ENHANCED: Load price history from API with moving averages
async function loadPriceHistory() {
    try {
        console.log('📈 Loading enhanced price history...');
        const response = await fetch('/api/price_history?days=60');
        const data = await response.json();
        
        if (data.status === 'success') {
            const prices = data.data;
            const metadata = data.metadata;
            console.log(`✅ Loaded ${prices.length} days of price data`);
            
            // Format data for time series chart
            const chartData = prices.map(item => ({
                x: new Date(item.date),
                y: item.price
            }));
            
            // Update price chart with enhanced data
            priceChart.data.datasets[0].data = chartData;
            
            // Clear existing datasets (keep only the main price line)
            while (priceChart.data.datasets.length > 1) {
                priceChart.data.datasets.pop();
            }
            
            // Add additional datasets for better visualization
            if (prices.length > 0) {
                // Calculate 7-day moving average
                const movingAvgData = calculateMovingAverage(prices, 7);
                priceChart.data.datasets.push({
                    label: '7-Day Moving Average',
                    data: movingAvgData,
                    borderColor: '#28a745',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0,
                    tension: 0.2
                });
                
                // Update chart title with market context
                updateChartTitle(metadata);
            }
            
            priceChart.update();
            
        } else {
            console.error('❌ Error in price history response:', data.message);
            loadSamplePriceData();
        }
    } catch (error) {
        console.error('❌ Error loading price history:', error);
        loadSamplePriceData();
    }
}

// Calculate moving average for trend line
function calculateMovingAverage(prices, period) {
    const movingAvg = [];
    for (let i = period - 1; i < prices.length; i++) {
        let sum = 0;
        for (let j = 0; j < period; j++) {
            sum += prices[i - j].price;
        }
        movingAvg.push({
            x: new Date(prices[i].date),
            y: sum / period
        });
    }
    return movingAvg;
}

// Update chart title with current market context
function updateChartTitle(metadata) {
    if (!metadata || !metadata.current_price) return;
    
    const currentPrice = metadata.current_price;
    const priceChange = metadata.price_change_24h;
    const marketStatus = metadata.market_status;
    
    const direction = priceChange.absolute >= 0 ? '📈' : '📉';
    const changeText = Math.abs(priceChange.percent).toFixed(2);
    const changeColor = priceChange.absolute >= 0 ? '#28a745' : '#dc3545';
    
    // Update the chart title
    priceChart.options.plugins.title.text = [
        `Bitcoin Price History - ${marketStatus.toUpperCase()} MARKET`,
        `${direction} $${currentPrice.toLocaleString()} (${priceChange.absolute >= 0 ? '+' : ''}${changeText}%)`
    ];
    
    // Update the title color based on direction
    priceChart.options.plugins.title.color = changeColor;
    priceChart.update();
    
    // Update any additional price indicators on the page
    updatePriceIndicators(metadata);
}

// Update additional price indicators
function updatePriceIndicators(metadata) {
    // Update any price display elements if they exist
    const priceElements = document.querySelectorAll('[data-price-indicator]');
    priceElements.forEach(element => {
        const format = element.getAttribute('data-price-format') || 'full';
        if (format === 'compact') {
            element.textContent = `$${(metadata.current_price / 1000).toFixed(0)}K`;
        } else {
            element.textContent = `$${metadata.current_price.toLocaleString()}`;
        }
    });
}

// ENHANCED: Load sentiment data from API
// ENHANCED: Load sentiment data from API
async function loadSentimentData() {
    try {
        console.log('📊 Loading sentiment data...');
        const response = await fetch('/api/sentiment_data');
        const data = await response.json();
        
        console.log('📊 Sentiment API response:', data);
        
        if (data.status === 'success') {
            const sentiment = data.data;
            
            // Validate data structure
            if (!sentiment || sentiment.positive === undefined || sentiment.neutral === undefined || sentiment.negative === undefined) {
                console.error('❌ Invalid sentiment data structure:', sentiment);
                // Use fallback data
                loadFallbackSentimentData();
                return;
            }
            
            console.log(`📊 Sentiment data: Positive=${sentiment.positive}, Neutral=${sentiment.neutral}, Negative=${sentiment.negative}`);
            
            // Update sentiment chart with the actual data
            sentimentChart.data.datasets[0].data = [
                sentiment.positive,
                sentiment.neutral,
                sentiment.negative
            ];
            
            // Update sentiment context
            updateSentimentContext(sentiment);
            
            // Update the chart
            sentimentChart.update();
            
            console.log('✅ Sentiment chart and context updated successfully');
        } else {
            console.error('❌ Error in sentiment API response:', data.message);
            loadFallbackSentimentData();
        }
    } catch (error) {
        console.error('❌ Error loading sentiment data:', error);
        loadFallbackSentimentData();
    }
}

// Fallback sentiment data
function loadFallbackSentimentData() {
    console.log('📊 Using fallback sentiment data');
    const fallbackData = {
        positive: 12, 
        neutral: 8, 
        negative: 5, 
        total_edits: 25, 
        avg_sentiment: 0.15, 
        sentiment_trend: 'improving',
        data_points: 25
    };
    
    sentimentChart.data.datasets[0].data = [
        fallbackData.positive,
        fallbackData.neutral,
        fallbackData.negative
    ];
    
    updateSentimentContext(fallbackData);
    sentimentChart.update();
}
// FIXED: Update sentiment context information - COMPLETE VERSION
// FIXED: Update sentiment context information - COMPLETE VERSION
function updateSentimentContext(sentiment) {
    console.log('📊 Updating sentiment context with:', sentiment);
    
    // Update sentiment insights section
    const totalEditsElement = document.getElementById('totalEdits');
    const avgSentimentElement = document.getElementById('avgSentiment');
    const sentimentTrendElement = document.getElementById('sentimentTrend');
    const sentimentDataPointsElement = document.getElementById('sentimentDataPoints');
    
    // Update total edits
    if (totalEditsElement) {
        totalEditsElement.textContent = sentiment.total_edits || 0;
    }
    
    // Update average sentiment
    if (avgSentimentElement) {
        avgSentimentElement.textContent = typeof sentiment.avg_sentiment === 'number' 
            ? sentiment.avg_sentiment.toFixed(2) 
            : '0.00';
    }
    
    // Update sentiment trend with proper formatting
    if (sentimentTrendElement) {
        const trend = sentiment.sentiment_trend || 'stable';
        const trendIcon = trend === 'improving' ? '📈' : 
        trend === 'declining' ? '📉' : '➡️';
        sentimentTrendElement.innerHTML = `${trendIcon} ${trend}`;
        sentimentTrendElement.className = `sentiment-trend ${trend}`;
    }
    
    // Update data points
    if (sentimentDataPointsElement) {
        sentimentDataPointsElement.textContent = sentiment.data_points || 0;
    }
    
    // Also update any individual sentiment elements if they exist
    const sentimentElements = {
        positive: document.getElementById('sentimentPositive'),
        neutral: document.getElementById('sentimentNeutral'),
        negative: document.getElementById('sentimentNegative')
    };
    
    // Update individual sentiment counts if elements exist
    if (sentimentElements.positive) {
        sentimentElements.positive.textContent = sentiment.positive || 0;
    }
    if (sentimentElements.neutral) {
        sentimentElements.neutral.textContent = sentiment.neutral || 0;
    }
    if (sentimentElements.negative) {
        sentimentElements.negative.textContent = sentiment.negative || 0;
    }
    
    console.log('✅ Sentiment context updated successfully');
}

// ENHANCED: Load performance data from API
// ENHANCED: Load performance data from API
async function loadPerformanceData() {
    try {
        const response = await fetch('/api/model_performance');
        const data = await response.json();
        
        if (data.status === 'success') {
            const performance = data.data;
            
            // FIXED: Update performance chart for backtest data
            if (performance.data_source === 'backtesting' || performance.data_source === 'backtesting_fallback') {
                // Use backtest accuracy for the chart
                const accuracy = performance.accuracy / 100; // Convert percentage to decimal
                performanceChart.data.datasets[0].data = [
                    accuracy * 100,  // Correct predictions (as percentage)
                    (1 - accuracy) * 100  // Incorrect predictions (as percentage)
                ];
            } else {
                // Use historical prediction data (original logic)
                performanceChart.data.datasets[0].data = [
                    performance.correct_predictions,
                    performance.total_predictions - performance.correct_predictions
                ];
            }
            
            performanceChart.update();
            
            // Update stats cards with enhanced information
            document.getElementById('accuracyStat').textContent = `${Math.round(performance.accuracy)}%`;
            document.getElementById('upAccuracy').textContent = `${Math.round(performance.up_accuracy)}%`;
            document.getElementById('downAccuracy').textContent = `${Math.round(performance.down_accuracy)}%`;
            document.getElementById('avgConfidence').textContent = `${Math.round(performance.avg_confidence)}%`;
            
            // Update performance context
            updatePerformanceContext(performance);
        }
    } catch (error) {
        console.error('❌ Error loading performance data:', error);
        // Fallback to sample data
        document.getElementById('accuracyStat').textContent = '65%';
        document.getElementById('upAccuracy').textContent = '68%';
        document.getElementById('downAccuracy').textContent = '62%';
        document.getElementById('avgConfidence').textContent = '71%';
        
        // Fallback for chart
        performanceChart.data.datasets[0].data = [65, 35];
        performanceChart.update();
    }
}

// Update performance context information
function updatePerformanceContext(performance) {
    const performanceElements = {
        grade: document.getElementById('performanceGrade'),
        trend: document.getElementById('performanceTrend'),
        quality: document.getElementById('confidenceQuality')
    };
    
    if (performanceElements.grade) {
        performanceElements.grade.textContent = performance.performance_grade;
        performanceElements.grade.className = `grade-${performance.performance_grade.toLowerCase()}`;
    }
    
    if (performanceElements.trend) {
        const trendIcon = performance.recent_trend === 'improving' ? '📈' : 
                        performance.recent_trend === 'declining' ? '📉' : '➡️';
        performanceElements.trend.innerHTML = `${trendIcon} ${performance.recent_trend}`;
    }
    
    if (performanceElements.quality) {
        performanceElements.quality.textContent = performance.confidence_quality;
        performanceElements.quality.className = `quality-${performance.confidence_quality.toLowerCase()}`;
    }
}

// Load analytics data
async function loadAnalyticsData() {
    console.log('📈 Loading analytics data...');
    await loadFeatureImportance();
    await loadConfidenceDistribution();
    
    // Force a resize to ensure proper layout
    setTimeout(() => {
        if (featureChart) {
            featureChart.resize();
            featureChart.update();
        }
        if (confidenceChart) {
            confidenceChart.resize();
            confidenceChart.update();
        }
    }, 100);
}

// FIXED: Enhanced feature importance loading with sample data fallback
async function loadFeatureImportance() {
    try {
        console.log('📊 Loading feature importance...');
        const response = await fetch('/api/feature_importance');
        const data = await response.json();
        
        console.log('📊 Feature importance API response:', data);
        
        if (data.status === 'success') {
            const features = data.data;
            
            // Update the feature chart
            if (features.features && features.importance) {
                featureChart.data.labels = features.features;
                featureChart.data.datasets[0].data = features.importance;
                featureChart.update();
            }
            
            // Update feature importance context with enhanced data
            updateFeatureContext(features);
            
        } else {
            console.warn('⚠️ Feature importance API returned error, using sample data');
            loadSampleFeatureData();
        }
    } catch (error) {
        console.error('❌ Error loading feature importance:', error);
        loadSampleFeatureData();
    }
}

// FIXED: Update feature importance context - WITH PROPER DATA HANDLING
function updateFeatureContext(features) {
    console.log('📊 Feature data received:', features);
    
    // FIX: Handle cases where categories might be missing or empty
    if (!features.categories) {
        // Auto-categorize features based on their names
        features.categories = autoCategorizeFeatures(features.features || []);
    }
    
    // Update category information
    const categoryElements = {
        price: document.getElementById('categoryPrice'),
        sentiment: document.getElementById('categorySentiment'),
        wikipedia: document.getElementById('categoryWikipedia'),
        technical: document.getElementById('categoryTechnical')
    };
    
    // Update each category count
    for (const [category, element] of Object.entries(categoryElements)) {
        if (element) {
            const count = features.categories[category] ? features.categories[category].length : 0;
            element.textContent = `${count} features`;
            
            // Add visual feedback for categories with features
            if (count > 0) {
                element.parentElement.style.background = getCategoryColor(category, true);
            }
        }
    }
    
    // Update top feature display
    const topFeatureElement = document.getElementById('topFeature');
    if (topFeatureElement) {
        if (features.top_feature) {
            topFeatureElement.textContent = `Top Feature: ${features.top_feature}`;
        } else if (features.features && features.importance && features.features.length > 0) {
            // Auto-detect top feature from importance scores
            const maxImportance = Math.max(...features.importance);
            const topIndex = features.importance.indexOf(maxImportance);
            const topFeatureName = features.features[topIndex] || 'Unknown Feature';
            topFeatureElement.textContent = `Top Feature: ${topFeatureName}`;
        } else {
            // Show sample data if no real data available
            topFeatureElement.textContent = `Top Feature: Price Momentum`;
            
            // Also update categories with sample data
            updateCategoriesWithSampleData();
        }
    }
}

// FIXED: Auto-categorize features based on feature names
function autoCategorizeFeatures(featureNames) {
    const categories = {
        price: [],
        sentiment: [],
        wikipedia: [],
        technical: []
    };
    
    if (!featureNames || featureNames.length === 0) {
        // Return sample categories if no features available
        return {
            price: ['close_ratio_5', 'close_ratio_20', 'price_momentum'],
            sentiment: ['sentiment_score', 'sentiment_trend'],
            wikipedia: ['edit_count_24h', 'edit_frequency'],
            technical: ['rsi', 'volume_trend', 'volatility']
        };
    }
    
    featureNames.forEach(feature => {
        const featureLower = feature.toLowerCase();
        
        if (featureLower.includes('close') || featureLower.includes('price') || featureLower.includes('ratio')) {
            categories.price.push(feature);
        } else if (featureLower.includes('sentiment') || featureLower.includes('emotion') || featureLower.includes('score')) {
            categories.sentiment.push(feature);
        } else if (featureLower.includes('edit') || featureLower.includes('wikipedia') || featureLower.includes('count')) {
            categories.wikipedia.push(feature);
        } else if (featureLower.includes('rsi') || featureLower.includes('volume') || featureLower.includes('trend') || featureLower.includes('momentum')) {
            categories.technical.push(feature);
        } else {
            // Default to technical for unrecognized features
            categories.technical.push(feature);
        }
    });
    
    return categories;
}

// Get category color with active state
function getCategoryColor(category, isActive = false) {
    const colors = {
        price: isActive ? '#d4edda' : '#e8f5e8',
        sentiment: isActive ? '#d1ecf1' : '#e8f4fd',
        wikipedia: isActive ? '#fff3cd' : '#fff3cd',
        technical: isActive ? '#f8d7da' : '#f8d7da'
    };
    return colors[category] || '#f8f9fa';
}

// Update categories with sample data when no real data is available
function updateCategoriesWithSampleData() {
    const categoryElements = {
        price: document.getElementById('categoryPrice'),
        sentiment: document.getElementById('categorySentiment'),
        wikipedia: document.getElementById('categoryWikipedia'),
        technical: document.getElementById('categoryTechnical')
    };
    
    const sampleCounts = {
        price: 3,
        sentiment: 2,
        wikipedia: 2,
        technical: 3
    };
    
    for (const [category, element] of Object.entries(categoryElements)) {
        if (element) {
            element.textContent = `${sampleCounts[category]} features`;
            element.parentElement.style.background = getCategoryColor(category, true);
        }
    }
}

// Load sample feature data when API fails
function loadSampleFeatureData() {
    console.log('📊 Loading sample feature data...');
    
    const sampleFeatures = {
        features: [
            'close_ratio_5', 'close_ratio_20', 'price_momentum', 
            'sentiment_score', 'sentiment_trend', 'edit_count_24h',
            'edit_frequency', 'rsi', 'volume_trend', 'volatility'
        ],
        importance: [0.15, 0.12, 0.18, 0.22, 0.08, 0.07, 0.05, 0.06, 0.04, 0.03],
        top_feature: 'sentiment_score'
    };
    
    // Update the chart with sample data
    featureChart.data.labels = sampleFeatures.features;
    featureChart.data.datasets[0].data = sampleFeatures.importance;
    featureChart.update();
    
    // Update the context with sample data
    updateFeatureContext(sampleFeatures);
    
    // Also update categories with sample data
    updateCategoriesWithSampleData();
}

// ENHANCED: Load confidence distribution (calculated from history)
async function loadConfidenceDistribution() {
    try {
        const response = await fetch('/api/prediction_history?limit=50');
        const data = await response.json();
        
        if (data.status === 'success') {
            const history = data.data;
            
            let highConfidence = 0, mediumConfidence = 0, lowConfidence = 0;
            
            history.forEach(prediction => {
                if (prediction.confidence >= 70) highConfidence++;
                else if (prediction.confidence >= 50) mediumConfidence++;
                else lowConfidence++;
            });
            
            confidenceChart.data.datasets[0].data = [highConfidence, mediumConfidence, lowConfidence];
            confidenceChart.update();
            
            // Update confidence distribution stats
            updateConfidenceStats(highConfidence, mediumConfidence, lowConfidence);
        }
    } catch (error) {
        console.error('❌ Error loading confidence distribution:', error);
    }
}

// Update confidence distribution statistics
function updateConfidenceStats(high, medium, low) {
    const total = high + medium + low;
    if (total === 0) return;
    
    const statsElement = document.getElementById('confidenceStats');
    if (statsElement) {
        statsElement.innerHTML = `
            <div>High Confidence: ${high} (${((high/total)*100).toFixed(1)}%)</div>
            <div>Medium Confidence: ${medium} (${((medium/total)*100).toFixed(1)}%)</div>
            <div>Low Confidence: ${low} (${((low/total)*100).toFixed(1)}%)</div>
        `;
    }
}

// FIXED: Load history data with proper layout
async function loadHistoryData() {
    await loadPredictionHistory();
    updateCalendarView();
    // Force a resize after loading to fix any layout issues
    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 100);
}

// ENHANCED: Load prediction history from API
async function loadPredictionHistory() {
    try {
        const response = await fetch('/api/prediction_history?limit=20');
        const data = await response.json();
        
        if (data.status === 'success') {
            predictionHistory = data.data;
            updateHistoryDisplay();
            
            // Update history metadata
            if (data.metadata) {
                updateHistoryMetadata(data.metadata);
            }
        }
    } catch (error) {
        console.error('❌ Error loading prediction history:', error);
        loadSampleHistory();
    }
}

// FIXED: Enhanced history display with better layout
function updateHistoryDisplay() {
    const historyList = document.getElementById('historyList');
    if (!historyList) return;
    
    historyList.innerHTML = '';
    
    if (predictionHistory.length === 0) {
        historyList.innerHTML = `
            <div class="no-data" style="text-align: center; padding: 40px; color: #6c757d;">
                <i class="fas fa-history" style="font-size: 3em; margin-bottom: 15px; display: block;"></i>
                <p>No prediction history available</p>
                <small>Make your first prediction to see history</small>
            </div>
        `;
        return;
    }
    
    // Create a container with proper spacing
    historyList.style.display = 'flex';
    historyList.style.flexDirection = 'column';
    historyList.style.gap = '12px';
    historyList.style.maxHeight = '400px';
    historyList.style.overflowY = 'auto';
    historyList.style.padding = '10px 5px';
    
    predictionHistory.forEach((item, index) => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.style.cssText = `
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: white;
            border-radius: 10px;
            border-left: 4px solid ${item.prediction === 'UP' ? '#28a745' : '#F7931A'};
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            margin-bottom: 0;
        `;
        
        historyItem.onmouseenter = function() {
            this.style.transform = 'translateY(-2px)';
            this.style.boxShadow = '0 4px 12px rgba(0,0,0,0.12)';
        };
        
        historyItem.onmouseleave = function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '0 2px 8px rgba(0,0,0,0.08)';
        };

        const date = new Date(item.timestamp);
        const formattedDate = date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
        const formattedTime = date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });
        
        // Enhanced probability display
        const upProb = Math.round(item.up_probability * 100) / 100;
        const downProb = Math.round(item.down_probability * 100) / 100;
        
        // Determine accuracy indicator if available
        let accuracyIndicator = '';
        if (item.correct !== null) {
            accuracyIndicator = item.correct ? 
                '<span class="accuracy-badge correct" title="Correct Prediction" style="margin-right: 8px;">✓</span>' :
                '<span class="accuracy-badge incorrect" title="Incorrect Prediction" style="margin-right: 8px;">✗</span>';
        }
        
        historyItem.innerHTML = `
            <div class="history-date" style="flex: 1; min-width: 120px;">
                <strong style="display: block; margin-bottom: 4px;">${formattedDate}</strong>
                <div class="history-time" style="font-size: 0.85em; color: #6c757d;">${formattedTime}</div>
                <div class="history-price" style="color: #F7931A; font-weight: bold; margin-top: 4px;">
                    $${item.current_price?.toLocaleString() || 'N/A'}
                </div>
            </div>
            <div class="history-prediction" style="flex: 1; text-align: right; min-width: 150px;">
                <div style="display: flex; align-items: center; justify-content: flex-end; margin-bottom: 8px;">
                    ${accuracyIndicator}
                    <span class="prediction-badge ${item.prediction === 'UP' ? 'badge-up' : 'badge-down'}" 
                          style="padding: 6px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600;">
                        ${item.prediction} (${item.confidence}%)
                    </span>
                </div>
                <div class="probability-breakdown" style="font-size: 0.8em;">
                    <span class="prob-up" style="color: #28a745; margin-right: 10px;">UP: ${upProb}%</span>
                    <span class="prob-down" style="color: #F7931A;">DOWN: ${downProb}%</span>
                </div>
            </div>
        `;
        
        historyList.appendChild(historyItem);
    });
    
    // Update performance summary
    updatePerformanceSummary();
}

// FIXED: Update history metadata with better styling
function updateHistoryMetadata(metadata) {
    const metadataElement = document.getElementById('historyMetadata');
    if (metadataElement) {
        metadataElement.style.cssText = `
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            font-size: 0.9em;
            border-left: 4px solid #17A2B8;
        `;
        
        if (metadata && metadata.recent_accuracy) {
            metadataElement.innerHTML = `
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div>
                        <strong>Recent Accuracy:</strong> 
                        <span style="color: #28a745; font-weight: bold;">${metadata.recent_accuracy}%</span>
                    </div>
                    <div>
                        <strong>Date Range:</strong> 
                        <span>${metadata.date_range?.oldest || 'N/A'} to ${metadata.date_range?.newest || 'N/A'}</span>
                    </div>
                </div>
            `;
        } else {
            metadataElement.innerHTML = `
                <div style="text-align: center; color: #6c757d;">
                    <i class="fas fa-info-circle"></i> 
                    Historical data will appear after multiple predictions
                </div>
            `;
        }
    }
}

// FIXED: Enhanced performance summary with better layout
function updatePerformanceSummary() {
    const summaryElement = document.getElementById('performanceSummary');
    if (!summaryElement) return;
    
    // Clear any existing content and reset styles
    summaryElement.innerHTML = '';
    summaryElement.style.cssText = `
        min-height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
    `;
    
    if (predictionHistory.length === 0) {
        summaryElement.innerHTML = `
            <div class="no-data" style="text-align: center; color: #6c757d; width: 100%;">
                <i class="fas fa-chart-line" style="font-size: 2.5em; margin-bottom: 10px; display: block;"></i>
                <p style="margin-bottom: 5px;">No performance data</p>
                <small>Make predictions to see performance metrics</small>
            </div>
        `;
        return;
    }
    
    const total = predictionHistory.length;
    const avgConfidence = predictionHistory.reduce((sum, item) => sum + item.confidence, 0) / total;
    
    // Calculate additional metrics
    const upPredictions = predictionHistory.filter(p => p.prediction === 'UP').length;
    const downPredictions = predictionHistory.filter(p => p.prediction === 'DOWN').length;
    const highConfidence = predictionHistory.filter(p => p.confidence >= 70).length;
    const correctPredictions = predictionHistory.filter(p => p.correct === true).length;
    const accuracy = total > 0 ? (correctPredictions / total) * 100 : 0;
    
    // Reset styles for content
    summaryElement.style.cssText = '';
    
    summaryElement.innerHTML = `
        <div class="performance-metrics" style="
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 0;
        ">
            <div class="metric" style="
                padding: 12px;
                background: #f8f9fa;
                border-radius: 8px;
                text-align: center;
                border-top: 4px solid #17A2B8;
            ">
                <div style="font-size: 1.8em; font-weight: bold; color: #333;">${total}</div>
                <div style="font-size: 0.85em; color: #6c757d;">Total Predictions</div>
            </div>
            
            <div class="metric" style="
                padding: 12px;
                background: #f8f9fa;
                border-radius: 8px;
                text-align: center;
                border-top: 4px solid #28a745;
            ">
                <div style="font-size: 1.8em; font-weight: bold; color: #333;">${Math.round(accuracy)}%</div>
                <div style="font-size: 0.85em; color: #6c757d;">Accuracy Rate</div>
            </div>
            
            <div class="metric" style="
                padding: 12px;
                background: #f8f9fa;
                border-radius: 8px;
                text-align: center;
                border-top: 4px solid #F7931A;
            ">
                <div style="font-size: 1.8em; font-weight: bold; color: #333;">${Math.round(avgConfidence)}%</div>
                <div style="font-size: 0.85em; color: #6c757d;">Avg Confidence</div>
            </div>
            
            <div class="metric" style="
                padding: 12px;
                background: #f8f9fa;
                border-radius: 8px;
                text-align: center;
                border-top: 4px solid #20c997;
            ">
                <div style="font-size: 1.8em; font-weight: bold; color: #333;">${highConfidence}</div>
                <div style="font-size: 0.85em; color: #6c757d;">High Confidence</div>
            </div>
            
            <div class="metric" style="
                padding: 12px;
                background: #f8f9fa;
                border-radius: 8px;
                text-align: center;
                border-top: 4px solid #28a745;
                grid-column: 1 / -1;
            ">
                <div style="display: flex; justify-content: space-around;">
                    <div>
                        <div style="font-size: 1.5em; font-weight: bold; color: #28a745;">${upPredictions}</div>
                        <div style="font-size: 0.8em; color: #6c757d;">UP Predictions</div>
                    </div>
                    <div>
                        <div style="font-size: 1.5em; font-weight: bold; color: #F7931A;">${downPredictions}</div>
                        <div style="font-size: 0.8em; color: #6c757d;">DOWN Predictions</div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// FIXED: Enhanced calendar view with better layout
function updateCalendarView() {
    const calendarGrid = document.getElementById('calendarGrid');
    if (!calendarGrid) return;
    
    // Reset styles
    calendarGrid.innerHTML = '';
    calendarGrid.style.cssText = `
        display: grid;
        grid-template-columns: repeat(7, 1fr);
        gap: 8px;
        margin-top: 15px;
    `;
    
    // Get last 7 days of predictions
    const lastWeek = predictionHistory.slice(0, 7);
    
    for (let i = 6; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        const dateStr = date.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric'
        });
        
        const prediction = lastWeek.find(p => {
            const predDate = new Date(p.timestamp);
            return predDate.toDateString() === date.toDateString();
        });
        
        const dayElement = document.createElement('div');
        const predictionClass = prediction ? 
            (prediction.prediction === 'UP' ? 'up' : 'down') : 
            'no-prediction';
            
        dayElement.className = `calendar-day ${predictionClass}`;
        dayElement.style.cssText = `
            text-align: center;
            padding: 12px 5px;
            border-radius: 8px;
            font-size: 0.8em;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 70px;
            transition: all 0.3s ease;
            ${
                predictionClass === 'up' ? 
                    'background: rgba(40, 167, 69, 0.1); color: #28a745; border: 1px solid rgba(40, 167, 69, 0.2);' :
                predictionClass === 'down' ? 
                    'background: rgba(247, 147, 26, 0.1); color: #F7931A; border: 1px solid rgba(247, 147, 26, 0.2);' :
                    'background: #f8f9fa; color: #6c757d; border: 1px solid #e9ecef;'
            }
        `;
        
        dayElement.onmouseenter = function() {
            this.style.transform = 'scale(1.05)';
            this.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';
        };
        
        dayElement.onmouseleave = function() {
            this.style.transform = 'scale(1)';
            this.style.boxShadow = 'none';
        };
        
        let confidenceBadge = '';
        if (prediction) {
            const confidenceLevel = prediction.confidence >= 70 ? 'high' : 
            prediction.confidence >= 50 ? 'medium' : 'low';
            confidenceBadge = `
                <div class="confidence-dot ${confidenceLevel}" style="
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    margin-top: 5px;
                    ${confidenceLevel === 'high' ? 'background: #28a745;' : 
                    confidenceLevel === 'medium' ? 'background: #F7931A;' : 
                    'background: #dc3545;'}
                "></div>
            `;
        }
        
        dayElement.innerHTML = `
            <div class="calendar-date" style="font-weight: bold; margin-bottom: 5px;">${dateStr}</div>
            <div class="calendar-prediction" style="font-size: 0.9em; font-weight: 600;">
                ${prediction ? prediction.prediction : '-'}
            </div>
            ${confidenceBadge}
        `;
        
        // Add tooltip for more info
        if (prediction) {
            dayElement.title = `${prediction.prediction} - ${prediction.confidence}% confidence - $${prediction.current_price?.toLocaleString() || 'N/A'}`;
        }
        
        calendarGrid.appendChild(dayElement);
    }
}

// ENHANCED: Update data freshness indicator
function updateDataFreshness() {
    const freshnessElement = document.getElementById('dataFreshness');
    if (!freshnessElement) return;
    
    // This would ideally come from the backend status
    const now = new Date();
    const lastUpdate = new Date(); // This should come from backend status
    const hoursSinceUpdate = Math.floor((now - lastUpdate) / (1000 * 60 * 60));
    
    let freshnessHTML = '';
    if (hoursSinceUpdate < 1) {
        freshnessHTML = '<span class="fresh">🟢 Data Updated < 1h ago</span>';
    } else if (hoursSinceUpdate < 24) {
        freshnessHTML = `<span class="stale">🟡 Data Updated ${hoursSinceUpdate}h ago</span>`;
    } else {
        freshnessHTML = '<span class="outdated">🔴 Data Outdated - Run Update</span>';
    }
    
    // Add additional freshness context
    freshnessHTML += `<br><small>Model trained with latest Wikipedia sentiment analysis</small>`;
    
    freshnessElement.innerHTML = freshnessHTML;
}

// ENHANCED: Fallback to sample data if APIs fail
function loadSamplePriceData() {
    console.log('⚠️ Using enhanced sample price data (fallback)');
    
    const samplePrices = [];
    const today = new Date();
    let basePrice = 85000; // More realistic starting price
    
    for (let i = 60; i >= 0; i--) {
        const date = new Date();
        date.setDate(today.getDate() - i);
        
        // More realistic price simulation with volatility
        const volatility = 0.02; // 2% daily volatility
        const change = (Math.random() - 0.5) * 2 * volatility * basePrice;
        const price = basePrice + change;
        
        samplePrices.push({
            date: date.toISOString().split('T')[0],
            price: Math.round(price)
        });
        
        basePrice = price;
    }
    
    // Format for time series chart
    const chartData = samplePrices.map(item => ({
        x: new Date(item.date),
        y: item.price
    }));
    
    priceChart.data.datasets[0].data = chartData;
    
    // Add moving average to sample data too
    const movingAvgData = calculateMovingAverage(samplePrices, 7);
    priceChart.data.datasets.push({
        label: '7-Day Moving Average',
        data: movingAvgData,
        borderColor: '#28a745',
        backgroundColor: 'rgba(40, 167, 69, 0.1)',
        borderWidth: 2,
        borderDash: [5, 5],
        fill: false,
        pointRadius: 0,
        tension: 0.2
    });
    
    priceChart.update();
    
    // Update title for sample data
    priceChart.options.plugins.title.text = [
        'Bitcoin Price History (Sample Data)',
        '📈 $' + Math.round(basePrice).toLocaleString() + ' (Demo Mode)'
    ];
    priceChart.update();
}

function loadSampleHistory() {
    const sampleHistory = [
        { 
            timestamp: new Date().toISOString(), 
            prediction: 'UP', 
            confidence: 68, 
            current_price: 95123, 
            up_probability: 68, 
            down_probability: 32,
            correct: true
        },
        { 
            timestamp: new Date(Date.now() - 86400000).toISOString(), 
            prediction: 'DOWN', 
            confidence: 72, 
            current_price: 94876, 
            up_probability: 28, 
            down_probability: 72,
            correct: false
        },
        { 
            timestamp: new Date(Date.now() - 172800000).toISOString(), 
            prediction: 'UP', 
            confidence: 54, 
            current_price: 94567, 
            up_probability: 54, 
            down_probability: 46,
            correct: true
        },
    ];
    
    predictionHistory = sampleHistory;
    updateHistoryDisplay();
    updateCalendarView();
}

// Enhanced getPrediction function with better loading states
async function getPrediction() {
    showLoading(
        'Analyzing current market data...',
        'Processing Wikipedia sentiment and technical indicators...'
    );
    disableButtons(true);
    
    try {
        console.log('🎯 Fetching prediction from server...');
        const response = await fetch('/predict');
        const data = await response.json();
        
        console.log('Prediction API response:', data);
        
        hideLoading();
        disableButtons(false);
        
        if (data.error) {
            console.error('❌ Prediction error from server:', data.error);
            showError(data.error);
            return;
        }
        
        // Ensure we have all required fields
        if (!data.prediction || data.confidence === undefined || !data.current_price) {
            console.error('❌ Invalid prediction data received:', data);
            showError('Invalid prediction data received from server');
            return;
        }
        
        console.log('✅ Updating display with valid prediction data');
        updatePredictionDisplay(data);
        checkStatus();
        updateDataFreshness();
        
        // Refresh dashboard data
        loadDashboardData();
        
    } catch (error) {
        console.error('❌ Network error fetching prediction:', error);
        hideLoading();
        disableButtons(false);
        showError('Failed to get prediction: ' + error.message);
    }
}

async function updateData() {
    showLoading(
        'Updating Wikipedia and price data...',
        'This may take a few minutes as we retrain the AI model...'
    );
    disableButtons(true);
    
    try {
        const response = await fetch('/update', { method: 'POST' });
        const data = await response.json();
        
        hideLoading();
        disableButtons(false);
        
        if (data.status === 'success') {
            // Show success message with animation
            const updateBtn = document.getElementById('updateBtn');
            updateBtn.innerHTML = '<i class="fas fa-check"></i> Updated!';
            updateBtn.style.background = 'linear-gradient(135deg, #28a745, #20c997)';
            
            setTimeout(() => {
                updateBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Update Data';
                updateBtn.style.background = '';
            }, 2000);
            
            checkStatus();
            updateDataFreshness();
            loadDashboardData();
        } else {
            showError('Update failed: ' + data.message);
        }
        
    } catch (error) {
        hideLoading();
        disableButtons(false);
        showError('Update failed: ' + error.message);
    }
}

// ENHANCED: Check status with more detailed information
async function checkStatus() {
    try {
        const response = await fetch('/status');
        const data = await response.json();
        
        let statusHTML = `
            Model: ${data.model_loaded ? '✅ Loaded' : '❌ Missing'}<br>
            Data: ${data.data_loaded ? '✅ Loaded' : '❌ Missing'}<br>
            Last Update: ${data.last_update}<br>
            Current Time: ${data.current_time}
        `;
        
        // Add system health information if available
        if (data.system_health) {
            const health = data.system_health;
            statusHTML += `<br>Health: <span class="health-${health.health_status}">${health.health_status.toUpperCase()} (${health.health_score}%)</span>`;
            statusHTML += `<br>Data Freshness: ${health.data_freshness}</div>`;
            statusHTML += `<br>Predictions: ${health.predictions_count}`;
        }
        
        document.getElementById('statusInfo').innerHTML = statusHTML;
    } catch (error) {
        console.error('❌ Status check failed:', error);
        document.getElementById('statusInfo').textContent = 'Unable to check status';
    }
}

// Enhanced prediction display function
function updatePredictionDisplay(data) {
    console.log('Starting prediction display update with:', data);
    
    // Get all required elements
    const card = document.getElementById('predictionCard');
    const predictionText = document.getElementById('predictionText');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    const priceText = document.getElementById('priceText');
    const dateText = document.getElementById('predictionDate');
    
    // Verify all elements exist
    if (!card || !predictionText || !confidenceValue || !confidenceFill || !priceText || !dateText) {
        console.error('❌ Missing required DOM elements:', {
            card: !!card,
            predictionText: !!predictionText,
            confidenceValue: !!confidenceValue,
            confidenceFill: !!confidenceFill,
            priceText: !!priceText,
            dateText: !!dateText
        });
        showError('UI elements not found');
        return;
    }
    
    try {
        // Reset card state
        card.className = 'prediction-card';
        card.classList.remove('pulse');
        
        // Set prediction style
        const isUp = data.prediction === 'UP';
        card.classList.add(isUp ? 'prediction-up' : 'prediction-down');
        
        // Update prediction text with enhanced styling
        predictionText.innerHTML = isUp ? 
            '<i class="fas fa-arrow-up"></i> PRICE WILL GO UP' : 
            '<i class="fas fa-arrow-down"></i> PRICE WILL GO DOWN';
        predictionText.style.color = isUp ? '#28a745' : '#F7931A';
        
        // Update confidence
        const confidence = parseFloat(data.confidence);
        confidenceValue.textContent = `${confidence}%`;
        
        // Update confidence meter with animation
        setTimeout(() => {
            confidenceFill.style.width = `${confidence}%`;
            
            // Set confidence color with enhanced visual feedback
            if (confidence >= 70) {
                confidenceFill.className = 'confidence-fill confidence-high';
                confidenceFill.style.background = 'linear-gradient(90deg, #28a745, #34ce57)';
            } else if (confidence >= 50) {
                confidenceFill.className = 'confidence-fill confidence-medium';
                confidenceFill.style.background = 'linear-gradient(90deg, #F7931A, #f9a93c)';
            } else {
                confidenceFill.className = 'confidence-fill confidence-low';
                confidenceFill.style.background = 'linear-gradient(90deg, #dc3545, #e74c3c)';
            }
        }, 100);
        
        // Update price with enhanced formatting
        const formattedPrice = new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 0,
            maximumFractionDigits: 0
        }).format(data.current_price);
        
        priceText.textContent = formattedPrice;
        priceText.classList.add('price-update');
        setTimeout(() => {
            priceText.classList.remove('price-update');
        }, 1500);
        
        // Update date with better formatting
        const predictionDate = new Date(data.prediction_date);
        const formattedDate = predictionDate.toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
        dateText.textContent = `Prediction for: ${formattedDate}`;
        
        // Add pulse animation for high confidence predictions
        if (confidence >= 80) {
            card.classList.add('pulse');
        }
        
        // Update probability breakdown if elements exist
        updateProbabilityBreakdown(data.prediction_proba);
        
        console.log('✅ Prediction display updated successfully');
        
    } catch (error) {
        console.error('❌ Error in prediction display:', error);
        showError('Display error: ' + error.message);
    }
}

// Update probability breakdown display
function updateProbabilityBreakdown(proba) {
    const upProbElement = document.getElementById('upProbability');
    const downProbElement = document.getElementById('downProbability');
    
    if (upProbElement && downProbElement) {
        upProbElement.textContent = `UP: ${proba.up_probability}%`;
        downProbElement.textContent = `DOWN: ${proba.down_probability}%`;
        
        // Add visual indicators for probability strength
        upProbElement.className = `probability ${proba.up_probability >= 60 ? 'high' : proba.up_probability >= 40 ? 'medium' : 'low'}`;
        downProbElement.className = `probability ${proba.down_probability >= 60 ? 'high' : proba.down_probability >= 40 ? 'medium' : 'low'}`;
    }
}

function showLoading(message, submessage = '') {
    const loading = document.getElementById('loading');
    const loadingText = document.getElementById('loadingText');
    const loadingSubtext = document.getElementById('loadingSubtext');
    
    if (loading && loadingText) {
        loading.style.display = 'block';
        loadingText.textContent = message;
        if (loadingSubtext && submessage) {
            loadingSubtext.textContent = submessage;
        }
    }
}

function hideLoading() {
    const loading = document.getElementById('loading');
    if (loading) {
        loading.style.display = 'none';
    }
}

function disableButtons(disabled) {
    const predictBtn = document.getElementById('predictBtn');
    const updateBtn = document.getElementById('updateBtn');
    if (predictBtn) predictBtn.disabled = disabled;
    if (updateBtn) updateBtn.disabled = disabled;
}

// Enhanced error display
function showError(message) {
    console.error('Showing error:', message);
    
    const predictionText = document.getElementById('predictionText');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    const priceText = document.getElementById('priceText');
    const dateText = document.getElementById('predictionDate');
    const card = document.getElementById('predictionCard');
    
    // Reset all elements to error state
    if (card) {
        card.className = 'prediction-card error-state';
        card.classList.remove('pulse', 'prediction-up', 'prediction-down');
    }
    
    if (predictionText) {
        predictionText.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Prediction Error';
        predictionText.style.color = '#dc3545';
    }
    
    if (confidenceValue) {
        confidenceValue.textContent = '0%';
    }
    
    if (confidenceFill) {
        confidenceFill.style.width = '0%';
        confidenceFill.className = 'confidence-fill confidence-low';
        confidenceFill.style.background = '#dc3545';
    }
    
    if (priceText) {
        priceText.textContent = '$--';
        priceText.style.color = '#6c757d';
    }
    
    if (dateText) {
        dateText.textContent = 'Unable to generate prediction';
        dateText.style.color = '#dc3545';
    }
    
    // Show error toast notification
    showErrorToast(message);
}

// Show error toast notification
function showErrorToast(message) {
    // Create toast element if it doesn't exist
    let toast = document.getElementById('errorToast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'errorToast';
        toast.className = 'error-toast';
        document.body.appendChild(toast);
    }
    
    toast.innerHTML = `
        <div class="toast-content">
            <i class="fas fa-exclamation-circle"></i>
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()">&times;</button>
        </div>
    `;
    
    toast.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        if (toast.parentElement) {
            toast.remove();
        }
    }, 5000);
}

// Debug function to check sentiment data
async function debugSentiment() {
    try {
        const response = await fetch('/api/debug_sentiment');
        const data = await response.json();
        console.log('🔧 Debug Sentiment Data:', data);
    } catch (error) {
        console.error('❌ Error debugging sentiment:', error);
    }
}

// Export prediction data for sharing
function exportPredictionData() {
    const data = {
        timestamp: new Date().toISOString(),
        predictions: predictionHistory,
        export_version: '1.0'
    };
    
    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `bitcoin-predictions-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

// Add CSS for new elements
function injectEnhancedStyles() {
    const styles = `
        .health-excellent { color: #28a745; font-weight: bold; }
        .health-healthy { color: #20c997; font-weight: bold; }
        .health-degraded { color: #fd7e14; font-weight: bold; }
        .health-poor { color: #dc3545; font-weight: bold; }
        .health-critical { color: #6f42c1; font-weight: bold; }
        
        .grade-a { color: #28a745; font-weight: bold; }
        .grade-b { color: #20c997; font-weight: bold; }
        .grade-c { color: #fd7e14; font-weight: bold; }
        .grade-d { color: #dc3545; font-weight: bold; }
        
        .quality-excellent { color: #28a745; }
        .quality-good { color: #20c997; }
        .quality-moderate { color: #fd7e14; }
        .quality-low { color: #dc3545; }
        
        .sentiment-trend.improving { color: #28a745; }
        .sentiment-trend.declining { color: #dc3545; }
        .sentiment-trend.stable { color: #6c757d; }
        
        .accuracy-badge {
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            text-align: center;
            line-height: 20px;
            font-size: 12px;
            margin-right: 5px;
        }
        
        .accuracy-badge.correct {
            background: #28a745;
            color: white;
        }
        
        .accuracy-badge.incorrect {
            background: #dc3545;
            color: white;
        }
        
        .probability-breakdown {
            font-size: 0.8em;
            margin-top: 5px;
        }
        
        .prob-up { color: #28a745; margin-right: 10px; }
        .prob-down { color: #F7931A; }
        
        .confidence-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-top: 2px;
        }
        
        .confidence-dot.high { background: #28a745; }
        .confidence-dot.medium { background: #F7931A; }
        .confidence-dot.low { background: #dc3545; }
        
        .error-toast {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #dc3545;
            color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 400px;
        }
        
        .toast-content {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .toast-content button {
            background: none;
            border: none;
            color: white;
            font-size: 18px;
            cursor: pointer;
        }
        
        .performance-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 15px;
        }
        
        .metric {
            padding: 8px;
            background: #f8f9fa;
            border-radius: 5px;
            font-size: 0.9em;
        }
        
        .no-data {
            text-align: center;
            padding: 20px;
            color: #6c757d;
        }
        
        .no-data i {
            font-size: 2em;
            margin-bottom: 10px;
            display: block;
        }
    `;
    
    const styleSheet = document.createElement('style');
    styleSheet.textContent = styles;
    document.head.appendChild(styleSheet);
}

// Uncomment the line below to debug sentiment data on page load
// debugSentiment();