// 全局变量
let trendChart = null;

// DOM 加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initializePage();
});

// 页面初始化
async function initializePage() {
    try {
        // 初始化日期选择器
        const today = new Date();
        const startDate = document.getElementById('startDate');
        const endDate = document.getElementById('endDate');
        
        if (startDate && endDate) {
            startDate.valueAsDate = new Date(today.getFullYear(), today.getMonth(), 1);
            endDate.valueAsDate = today;
            
            // 添加日期变化事件监听
            startDate.addEventListener('change', () => loadReportData());
            endDate.addEventListener('change', () => loadReportData());
        }

        // 初始化设备选择器
        const deviceBtns = document.querySelectorAll('.device-btn');
        deviceBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                deviceBtns.forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                loadReportData();
            });
        });

        // 初始化图表
        initializeTrendChart();

        // 加载初始数据
        await loadReportData();

        // 初始化加载更多按钮
        const loadMoreBtn = document.getElementById('loadMoreBtn');
        if (loadMoreBtn) {
            loadMoreBtn.addEventListener('click', loadMoreData);
        }

    } catch (error) {
        console.error('Page initialization error:', error);
        showNotification('页面初始化失败，请刷新重试', 'error');
    }
}

// 初始化趋势图表
function initializeTrendChart() {
    const canvas = document.getElementById('trendChart');
    if (!canvas) {
        console.error('Chart canvas element not found');
        return;
    }

    const ctx = canvas.getContext('2d');
    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: '发电量',
                data: [],
                borderColor: '#2196f3',
                tension: 0.4,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        drawBorder: false
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// 加载报表数据
async function loadReportData() {
    try {
        showLoading();
        
        // 模拟API调用
        const data = await fetchReportData();
        
        // 更新卡片数据
        updateReportCards(data);
        
        // 更新图表数据
        updateTrendChart(data);
        
        // 更新表格数据
        updateTableData(data);
        
        hideLoading();
    } catch (error) {
        console.error('Error loading report data:', error);
        showNotification('数据加载失败，请重试', 'error');
        hideLoading();
    }
}

// 模拟获取报表数据
async function fetchReportData() {
    // 模拟API延迟
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    return {
        runningHours: 156.8,
        runningHoursChange: 5.2,
        powerGeneration: 2458.6,
        powerGenerationChange: 12.3,
        faultCount: 3,
        faultCountChange: -25.0,
        trend: generateTrendData(),
        details: generateDetailsData()
    };
}

// 生成趋势数据
function generateTrendData() {
    const dates = [];
    const values = [];
    const today = new Date();
    
    for (let i = 6; i >= 0; i--) {
        const date = new Date(today);
        date.setDate(date.getDate() - i);
        dates.push(date.toLocaleDateString());
        values.push(Math.random() * 1000 + 1500);
    }
    
    return { dates, values };
}

// 生成详细数据
function generateDetailsData() {
    const devices = ['光伏设备-A1', '风机设备-B2', '储能设备-C3'];
    const data = [];
    
    for (let i = 0; i < 10; i++) {
        data.push({
            date: new Date(new Date().setDate(new Date().getDate() - i)).toLocaleDateString(),
            device: devices[Math.floor(Math.random() * devices.length)],
            runningHours: (Math.random() * 24).toFixed(1),
            powerGeneration: (Math.random() * 1000 + 500).toFixed(1),
            faultCount: Math.floor(Math.random() * 3)
        });
    }
    
    return data;
}

// 更新报表卡片
function updateReportCards(data) {
    const cards = {
        runningHours: { value: data.runningHours, change: data.runningHoursChange },
        powerGeneration: { value: data.powerGeneration, change: data.powerGenerationChange },
        faultCount: { value: data.faultCount, change: data.faultCountChange }
    };
    
    Object.entries(cards).forEach(([key, { value, change }]) => {
        const card = document.querySelector(`.report-card .main-value[data-type="${key}"]`);
        const changeEl = document.querySelector(`.report-card .sub-value[data-type="${key}"]`);
        
        if (card) card.textContent = value.toLocaleString();
        if (changeEl) {
            const icon = change >= 0 ? 'mdi-arrow-up' : 'mdi-arrow-down';
            const cls = change >= 0 ? 'up' : 'down';
            changeEl.innerHTML = `<i class="mdi ${icon}"></i>${Math.abs(change)}%`;
            changeEl.className = `sub-value ${cls}`;
        }
    });
}

// 更新趋势图表
function updateTrendChart(data) {
    if (!trendChart) return;
    
    trendChart.data.labels = data.trend.dates;
    trendChart.data.datasets[0].data = data.trend.values;
    trendChart.update();
}

// 更新表格数据
function updateTableData(data) {
    const tbody = document.getElementById('reportTableBody');
    if (!tbody) return;
    
    tbody.innerHTML = data.details.map(item => `
        <tr>
            <td>${item.date}</td>
            <td>${item.device}</td>
            <td>${item.runningHours}</td>
            <td>${item.powerGeneration}</td>
            <td>${item.faultCount}</td>
        </tr>
    `).join('');
}

// 加载更多数据
function loadMoreData() {
    // 实现加载更多逻辑
}

// 显示加载状态
function showLoading() {
    document.body.classList.add('loading');
}

// 隐藏加载状态
function hideLoading() {
    document.body.classList.remove('loading');
}

// 显示通知
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <i class="mdi mdi-${type === 'error' ? 'alert-circle' : 'information'}"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(notification);
    setTimeout(() => notification.classList.add('show'), 100);
    
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// 刷新数据
function refreshData() {
    loadReportData();
} 