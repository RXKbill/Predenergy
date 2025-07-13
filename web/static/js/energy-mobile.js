// 移动端能源系统脚本

// 定义全局图表配置
const commonChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
        mode: 'index',
        intersect: false,
    },
    plugins: {
        legend: {
            position: 'top',
            labels: {
                usePointStyle: true,
                padding: 15,
                font: { size: 12 }
            }
        },
        tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            titleColor: '#000',
            bodyColor: '#666',
            borderColor: '#ddd',
            borderWidth: 1,
            padding: 10,
            boxPadding: 5,
            usePointStyle: true,
            position: 'nearest'
        }
    },
    layout: {
        padding: {
            top: 20,
            right: 20,
            bottom: 20,
            left: 20
        }
    },
    scales: {
        y: {
            beginAtZero: false,
            grid: {
                drawBorder: false,
                color: 'rgba(0, 0, 0, 0.05)'
            },
            ticks: {
                font: { size: 11 },
                padding: 10
            }
        },
        x: {
            grid: {
                display: false
            },
            ticks: {
                font: { size: 11 },
                padding: 10
            }
        }
    },
    animation: {
        duration: 300,
        easing: 'easeInOutQuad'
    }
};

// 图表配置和初始化
document.addEventListener('DOMContentLoaded', async function() {
    try {
        // 初始化所有图表
        const charts = {
            power: initPowerPredictionChart(),
            load: initLoadPredictionChart(),
            price: initPricePredictionChart(),
            charging: initChargingPredictionChart()
        };
        
        // 初始化图表切换功能
        initChartSwitching();
        
        // 初始化其他功能
        initAlertHandling();
        enhanceChartInteractions();
        
        // 初始化新增功能
        initTimeRangeControl();
        initChartOptions();
    } catch (error) {
        console.error('Initialization error:', error);
    }
});

// 初始化卡片点击事件
function initCardClickEvents() {
    document.querySelectorAll('.prediction-card').forEach(card => {
        card.addEventListener('click', function(e) {
            // 如果点击的是按钮，不触发卡片点击事件
            if (e.target.matches('button') || e.target.closest('button')) {
                return;
            }
            
            const type = this.id.replace('PredictionCard', '').toLowerCase();
            const chart = Chart.getChart(`${type}PredictionChart`);
            if (chart) {
                const latestData = chart.data.datasets[0].data;
                showDetailPopup(type, latestData.length - 1, latestData[latestData.length - 1]);
            }
        });
    });
}

// 发电预测图表初始化
function initPowerPredictionChart() {
    const ctx = document.getElementById('powerPredictionChart').getContext('2d');
    
    // 生成更真实的光伏发电数据
    const solarData = [
        15,    // 6:00 - 日出，发电开始
        120,   // 8:00 - 快速上升
        220,   // 10:00 - 接近峰值
        280,   // 12:00 - 正午峰值
        240,   // 14:00 - 略有下降
        180,   // 16:00 - 明显下降
        90,    // 18:00 - 日落前
        20     // 20:00 - 日落，发电结束
    ];

    // 生成更真实的风电数据
    const windData = [
        150,   // 6:00 - 清晨风力较大
        130,   // 8:00 - 略有减弱
        100,   // 10:00 - 持续减弱
        85,    // 12:00 - 正午风力最小
        95,    // 14:00 - 开始增强
        140,   // 16:00 - 显著增强
        180,   // 18:00 - 傍晚风力增大
        200    // 20:00 - 夜间风力最大
    ];

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00'],
            datasets: [{
                label: '光伏发电预测',
                data: solarData,
                borderColor: '#ffc107',
                backgroundColor: 'rgba(255, 193, 7, 0.2)',
                borderWidth: 2,
                tension: 0.3,
                fill: true,
                pointRadius: 4,
                pointHoverRadius: 6
            }, {
                label: '风电预测',
                data: windData,
                borderColor: '#20c997',
                backgroundColor: 'rgba(32, 201, 151, 0.2)',
                borderWidth: 2,
                tension: 0.3,
                fill: true,
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            ...commonChartOptions,
            plugins: {
                ...commonChartOptions.plugins,
                tooltip: {
                    ...commonChartOptions.plugins.tooltip,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y} MW`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    suggestedMax: 300,
                    title: {
                        display: true,
                        text: '发电量 (MW)'
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                        drawBorder: false
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        }
    });
}

// 负荷预测图表初始化
function initLoadPredictionChart() {
    const ctx = document.getElementById('loadPredictionChart').getContext('2d');
    
    // 生成更真实的负荷数据
    const loadData = [
        160,   // 6:00 - 清晨用电开始增加
        240,   // 8:00 - 早高峰
        210,   // 10:00 - 早高峰后回落
        185,   // 12:00 - 午间低谷
        220,   // 14:00 - 下午用电上升
        260,   // 16:00 - 工业用电高峰
        280,   // 18:00 - 晚高峰
        250    // 20:00 - 晚高峰后期
    ];

    // 添加安全阈值数据
    const safetyThreshold = Array(8).fill(300);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00'],
            datasets: [{
                label: '负荷预测',
                data: loadData,
                borderColor: '#0dcaf0',
                backgroundColor: 'rgba(13, 202, 240, 0.2)',
                borderWidth: 2,
                tension: 0.3,
                fill: true,
                pointRadius: 4,
                pointHoverRadius: 6
            }, {
                label: '安全阈值',
                data: safetyThreshold,
                borderColor: '#dc3545',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false
            }]
        },
        options: {
            ...commonChartOptions,
            plugins: {
                ...commonChartOptions.plugins,
                tooltip: {
                    ...commonChartOptions.plugins.tooltip,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y} MW`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 150,
                    max: 320,
                    title: {
                        display: true,
                        text: '负荷 (MW)'
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                        drawBorder: false
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        }
    });
}

// 电价预测图表初始化
function initPricePredictionChart() {
    const ctx = document.getElementById('pricePredictionChart').getContext('2d');
    
    // 生成更真实的电价数据
    const priceData = [
        0.45,   // 6:00 - 谷电价
        0.68,   // 8:00 - 早高峰价格
        0.58,   // 10:00 - 平段电价
        0.55,   // 12:00 - 午间平段
        0.62,   // 14:00 - 下午平段
        0.75,   // 16:00 - 工业高峰
        0.85,   // 18:00 - 晚高峰最高
        0.72    // 20:00 - 高峰回落
    ];

    // 添加基准电价线
    const basePrice = Array(8).fill(0.60);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00'],
            datasets: [{
                label: '电价预测',
                data: priceData,
                borderColor: '#198754',
                backgroundColor: 'rgba(25, 135, 84, 0.2)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 4,
                pointHoverRadius: 6
            }, {
                label: '基准电价',
                data: basePrice,
                borderColor: '#6c757d',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false
            }]
        },
        options: {
            ...commonChartOptions,
            plugins: {
                ...commonChartOptions.plugins,
                tooltip: {
                    ...commonChartOptions.plugins.tooltip,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)} ¥/kWh`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 0.4,
                    max: 0.9,
                    title: {
                        display: true,
                        text: '电价 (¥/kWh)'
                    },
                    ticks: {
                        callback: value => value.toFixed(2)
                    }
                }
            }
        }
    });
}

// 充电预测图表初始化
function initChargingPredictionChart() {
    const ctx = document.getElementById('chargingPredictionChart').getContext('2d');
    
    // 生成更真实的充电需求数据
    const chargingData = [
        25,    // 6:00 - 早晨充电开始
        85,    // 8:00 - 上班充电高峰
        60,    // 10:00 - 工作时段
        45,    // 12:00 - 午休时段
        55,    // 14:00 - 下午工作时段
        75,    // 16:00 - 下班前充电
        95,    // 18:00 - 下班高峰
        70     // 20:00 - 夜间充电
    ];

    // 添加充电站容量上限
    const capacityLimit = Array(8).fill(100);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00'],
            datasets: [{
                label: '充电需求预测',
                data: chargingData,
                borderColor: '#fd7e14',
                backgroundColor: 'rgba(253, 126, 20, 0.2)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 4,
                pointHoverRadius: 6
            }, {
                label: '充电站容量',
                data: capacityLimit,
                borderColor: '#6c757d',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false
            }]
        },
        options: {
            ...commonChartOptions,
            plugins: {
                ...commonChartOptions.plugins,
                tooltip: {
                    ...commonChartOptions.plugins.tooltip,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y} kW`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 120,
                    title: {
                        display: true,
                        text: '充电功率 (kW)'
                    }
                }
            }
        }
    });
}

// 系统指标初始化
function initSystemMetrics() {
    // CPU使用率图表
    const cpuCtx = document.getElementById('cpuUsageChart').getContext('2d');
    new Chart(cpuCtx, {
        type: 'line',
        data: {
            labels: Array.from({length: 12}, (_, i) => `${i*5}min`),
            datasets: [{
                label: 'CPU使用率',
                data: generateRandomData(12, 40, 80),
                borderColor: '#4F8CBE',
                tension: 0.4
            }]
        },
        options: {
            ...commonChartOptions,
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// 简化的通知函数
function showNotification(message, type = 'info') {
    // 创建通知元素
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show`;
    notification.role = 'alert';
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.zIndex = '9999';
    notification.style.minWidth = '250px';
    
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // 添加到文档
    document.body.appendChild(notification);
    
    // 自动关闭
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}

// 处理告警
function handleAlert(alertItem, alertTitle) {
    // 获取告警详细信息
    const alertType = alertItem.classList.contains('urgent') ? 'urgent' : 'warning';
    const alertMessage = alertItem.querySelector('p').textContent;
    const alertTime = alertItem.querySelector('small').textContent;

    // 创建处理弹窗
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">${alertType === 'urgent' ? '紧急告警处理' : '告警处理'}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="alert-info mb-3">
                        <div class="d-flex justify-content-between align-items-start">
                            <h6 class="text-${alertType === 'urgent' ? 'danger' : 'warning'} mb-2">${alertTitle}</h6>
                            <span class="badge bg-${alertType === 'urgent' ? 'danger' : 'warning'}">${alertType === 'urgent' ? '紧急' : '警告'}</span>
                        </div>
                        <p class="mb-2">${alertMessage}</p>
                        <small class="text-muted">发生时间：${alertTime}</small>
                    </div>

                    <form id="alertHandleForm">
                        <div class="mb-3">
                            <label class="form-label">处理方式</label>
                            <select class="form-select" id="handleMethod" required>
                                <option value="">请选择处理方式...</option>
                                <option value="auto">自动调节</option>
                                <option value="manual">人工干预</option>
                                <option value="monitor">持续监控</option>
                            </select>
                        </div>

                        <div class="mb-3">
                            <label class="form-label">处理措施</label>
                            <textarea class="form-control" id="handleMeasures" rows="3" 
                                placeholder="请详细描述处理措施..." required></textarea>
                        </div>

                        <div class="mb-3">
                            <label class="form-label">预计完成时间</label>
                            <input type="datetime-local" class="form-control" id="estimatedTime" required>
                        </div>

                        <div class="mb-3">
                            <label class="form-label">处理级别</label>
                            <div class="btn-group w-100" role="group">
                                <input type="radio" class="btn-check" name="priority" id="priority1" value="high" required>
                                <label class="btn btn-outline-danger" for="priority1">高优先级</label>

                                <input type="radio" class="btn-check" name="priority" id="priority2" value="medium">
                                <label class="btn btn-outline-warning" for="priority2">中优先级</label>

                                <input type="radio" class="btn-check" name="priority" id="priority3" value="low">
                                <label class="btn btn-outline-success" for="priority3">低优先级</label>
                            </div>
                        </div>

                        <div class="form-check mb-3">
                            <input class="form-check-input" type="checkbox" id="notifyRelated">
                            <label class="form-check-label" for="notifyRelated">
                                通知相关人员
                            </label>
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
                    <button type="button" class="btn btn-primary" onclick="submitAlertHandle(this)">确认处理</button>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modal);
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();

    // 设置默认的预计完成时间（当前时间后2小时）
    const defaultTime = new Date();
    defaultTime.setHours(defaultTime.getHours() + 2);
    document.getElementById('estimatedTime').value = defaultTime.toISOString().slice(0, 16);

    // 监听模态框关闭事件进行清理
    modal.addEventListener('hidden.bs.modal', () => {
        document.body.removeChild(modal);
    });
}

// 提交告警处理
async function submitAlertHandle(submitBtn) {
    const form = document.getElementById('alertHandleForm');
    if (!form.checkValidity()) {
        form.reportValidity();
        return;
    }

    // 禁用提交按钮防止重复提交
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>处理中...';

    try {
        const handleData = {
            method: document.getElementById('handleMethod').value,
            measures: document.getElementById('handleMeasures').value,
            estimatedTime: document.getElementById('estimatedTime').value,
            priority: document.querySelector('input[name="priority"]:checked').value,
            notifyRelated: document.getElementById('notifyRelated').checked
        };

        // 模拟API调用
        await new Promise(resolve => setTimeout(resolve, 1500));

        // 更新告警状态
        const alertItem = document.querySelector('.alert-item.processing');
        if (alertItem) {
            // 添加处理状态标记
            const statusBadge = document.createElement('span');
            statusBadge.className = `badge bg-${handleData.priority === 'high' ? 'danger' : 
                                              handleData.priority === 'medium' ? 'warning' : 'success'} ms-2`;
            statusBadge.textContent = '处理中';
            alertItem.querySelector('h6').appendChild(statusBadge);

            // 添加处理信息
            const handleInfo = document.createElement('div');
            handleInfo.className = 'alert-handle-info mt-2 small';
            handleInfo.innerHTML = `
                <div class="text-muted">处理方式: ${getHandleMethodText(handleData.method)}</div>
                <div class="text-muted">预计完成: ${new Date(handleData.estimatedTime).toLocaleString()}</div>
            `;
            alertItem.querySelector('p').after(handleInfo);

            // 更新按钮状态
            alertItem.querySelectorAll('.btn').forEach(btn => btn.disabled = true);
        }

        // 关闭模态框
        const modal = submitBtn.closest('.modal');
        bootstrap.Modal.getInstance(modal).hide();

        // 显示成功通知
        showNotification('告警已开始处理，将持续跟踪处理状态', 'success');
        
        // 更新告警数量
        updateAlertCount();

    } catch (error) {
        console.error('Error handling alert:', error);
        showNotification('处理失败，请重试', 'error');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '确认处理';
    }
}

// 获取处理方式文本
function getHandleMethodText(method) {
    const methods = {
        'auto': '自动调节',
        'manual': '人工干预',
        'monitor': '持续监控'
    };
    return methods[method] || method;
}

// 查看告警详情
function viewAlertDetails(alertItem, alertTitle) {
    const alertType = alertItem.classList.contains('urgent') ? 'urgent' : 'warning';
    const alertMessage = alertItem.querySelector('p').textContent;
    const alertTime = alertItem.querySelector('small').textContent;

    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.innerHTML = `
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">告警详情</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="alert-info mb-4">
                        <div class="d-flex justify-content-between align-items-start mb-3">
                            <h6 class="text-${alertType === 'urgent' ? 'danger' : 'warning'} mb-0">${alertTitle}</h6>
                            <span class="badge bg-${alertType === 'urgent' ? 'danger' : 'warning'}">
                                ${alertType === 'urgent' ? '紧急' : '警告'}
                            </span>
                        </div>
                        <p class="mb-3">${alertMessage}</p>
                        <div class="alert-meta text-muted small">
                            <div>告警时间: ${alertTime}</div>
                            <div>告警ID: ${generateAlertId()}</div>
                        </div>
                    </div>
                    
                    <div class="alert-analysis mb-4">
                        <h6 class="mb-3">告警分析</h6>
                        <div class="analysis-charts row">
                            <div class="col-md-6 mb-3">
                                <canvas id="alertTrendChart"></canvas>
                            </div>
                            <div class="col-md-6 mb-3">
                                <canvas id="alertImpactChart"></canvas>
                            </div>
                        </div>
                        <div class="analysis-text mt-3">
                            <h6 class="mb-2">可能原因</h6>
                            <ul class="list-unstyled">
                                <li>• 设备负载突增</li>
                                <li>• 系统响应延迟</li>
                                <li>• 外部环境影响</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div class="alert-suggestions">
                        <h6 class="mb-3">处理建议</h6>
                        <div class="suggestion-list">
                            <div class="suggestion-item mb-2 p-2 bg-light rounded">
                                <strong>立即执行：</strong>检查系统负载状态
                            </div>
                            <div class="suggestion-item mb-2 p-2 bg-light rounded">
                                <strong>建议操作：</strong>调整负载分配策略
                            </div>
                            <div class="suggestion-item mb-2 p-2 bg-light rounded">
                                <strong>预防措施：</strong>优化系统监控阈值
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">关闭</button>
                    <button type="button" class="btn btn-primary" onclick="handleAlert(document.querySelector('.alert-item.selected'))">
                        立即处理
                    </button>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modal);
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();

    // 初始化告警趋势图表
    initAlertTrendChart();
    initAlertImpactChart();

    // 监听模态框关闭事件进行清理
    modal.addEventListener('hidden.bs.modal', () => {
        document.body.removeChild(modal);
    });
}

// 生成告警ID
function generateAlertId() {
    return 'ALT-' + Date.now().toString(36).toUpperCase();
}

// 初始化告警趋势图表
function initAlertTrendChart() {
    const ctx = document.getElementById('alertTrendChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['10分钟前', '8分钟前', '6分钟前', '4分钟前', '2分钟前', '现在'],
            datasets: [{
                label: '负载趋势',
                data: [75, 82, 89, 95, 98, 102],
                borderColor: '#dc3545',
                tension: 0.4,
                fill: false
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: '负载变化趋势'
                }
            }
        }
    });
}

// 初始化告警影响图表
function initAlertImpactChart() {
    const ctx = document.getElementById('alertImpactChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['系统性能', '用户体验', '能源效率', '设备寿命'],
            datasets: [{
                label: '影响程度',
                data: [85, 65, 45, 30],
                backgroundColor: [
                    'rgba(220, 53, 69, 0.8)',
                    'rgba(255, 193, 7, 0.8)',
                    'rgba(40, 167, 69, 0.8)',
                    'rgba(13, 202, 240, 0.8)'
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: '影响评估'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

// 更新告警处理初始化函数
function initAlertHandling() {
    document.querySelectorAll('.alert-item .btn').forEach(button => {
        button.addEventListener('click', function() {
            const alertItem = this.closest('.alert-item');
            const action = this.textContent.trim();
            const alertTitle = alertItem.querySelector('h6').textContent;
            
            // 标记当前处理的告警
            document.querySelectorAll('.alert-item').forEach(item => {
                item.classList.remove('selected', 'processing');
            });
            alertItem.classList.add('selected');
            
            if (action === '立即处理') {
                alertItem.classList.add('processing');
                handleAlert(alertItem, alertTitle);
            } else if (action === '查看详情') {
                viewAlertDetails(alertItem, alertTitle);
            } else if (action === '忽略') {
                dismissAlert(alertItem, alertTitle);
            }
        });
    });

    updateAlertCount();
}

// 忽略告警
function dismissAlert(alertItem, alertTitle) {
    alertItem.classList.add('fade-out');
    setTimeout(() => {
        alertItem.remove();
        updateAlertCount();
        showNotification(`已忽略告警: ${alertTitle}`, 'warning');
    }, 300);
}

// 更新未处理告警数量
function updateAlertCount() {
    const alertCount = document.querySelectorAll('.alert-item').length;
    const alertBadge = document.querySelector('.card-header .badge');
    const notificationBadge = document.querySelector('.notification-badge');
    
    if (alertBadge) {
        alertBadge.textContent = `${alertCount} 条未处理`;
    }
    
    if (notificationBadge) {
        notificationBadge.textContent = alertCount;
        notificationBadge.style.display = alertCount > 0 ? '' : 'none';
    }
}

// 添加新告警
function addNewAlert(alert) {
    const alertTimeline = document.querySelector('.alert-timeline');
    if (!alertTimeline) return;

    const alertItem = document.createElement('div');
    alertItem.className = `alert-item ${alert.type}`;
    alertItem.innerHTML = `
        <div class="d-flex justify-content-between mb-2">
            <h6 class="text-${alert.type === 'urgent' ? 'danger' : 'warning'} mb-0">${alert.title}</h6>
            <small class="text-muted">${alert.time}</small>
        </div>
        <p class="mb-2">${alert.message}</p>
        <div class="d-flex">
            <button class="btn btn-sm btn-${alert.type === 'urgent' ? 'danger' : 'warning'} me-2">
                ${alert.type === 'urgent' ? '立即处理' : '查看详情'}
            </button>
            <button class="btn btn-sm btn-light">忽略</button>
        </div>
    `;

    alertTimeline.insertBefore(alertItem, alertTimeline.firstChild);
    updateAlertCount();
    
    // 绑定新添加的告警按钮事件
    const buttons = alertItem.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            const action = this.textContent.trim();
            if (action === '立即处理' || action === '查看详情') {
                handleAlert(alertItem, alert.title);
            } else if (action === '忽略') {
                dismissAlert(alertItem, alert.title);
            }
        });
    });
}

// 模拟定期添加新告警
function simulateNewAlerts() {
    const alerts = [
        {
            type: 'urgent',
            title: '负荷超限预警',
            message: '预计下一小时负荷将超过安全阈值',
            time: '刚刚'
        },
        {
            type: 'warning',
            title: '设备异常提醒',
            message: '光伏组件3号运行效率低于阈值',
            time: '刚刚'
        }
    ];

    // 随机选择一个告警添加
    const randomAlert = alerts[Math.floor(Math.random() * alerts.length)];
    addNewAlert(randomAlert);
}

// 页面加载完成后初始化告警功能
document.addEventListener('DOMContentLoaded', function() {
    initializeAlerts();
    
    // 每隔5分钟模拟产生新告警
    setInterval(simulateNewAlerts, 300000);
});

// 添加告警相关的样式
const style = document.createElement('style');
style.textContent = `
    .alert-item {
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 8px;
        background: #fff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: opacity 0.3s ease, transform 0.3s ease;
    }
    
    .alert-item.fade-out {
        opacity: 0;
        transform: translateX(-20px);
    }
    
    .alert-item.urgent {
        border-left: 4px solid #dc3545;
    }
    
    .alert-item.warning {
        border-left: 4px solid #ffc107;
    }
    
    .notification-badge {
        position: absolute;
        top: -5px;
        right: -5px;
        min-width: 18px;
        height: 18px;
        padding: 0 5px;
        border-radius: 9px;
        background: #dc3545;
        color: #fff;
        font-size: 12px;
        line-height: 18px;
        text-align: center;
    }
`;
document.head.appendChild(style);

// 通知系统
class NotificationSystem {
    constructor() {
        this.container = document.getElementById('notification-container');
        this.notifications = new Map();
        this.counter = 0;
    }

    show(message, type = 'info', duration = 5000) {
        const id = this.counter++;
        const notification = this.createNotification(id, message, type);
        
        this.container.appendChild(notification);
        this.notifications.set(id, notification);
        
        // 强制重排以触发动画
        notification.offsetHeight;
        notification.classList.add('show');
        
        // 添加进度条动画
        const progressBar = notification.querySelector('.notification-progress-bar');
        progressBar.style.transition = `width ${duration}ms linear`;
        
        setTimeout(() => {
            progressBar.style.width = '0%';
        }, 50);

        // 设置自动关闭
        const timer = setTimeout(() => {
            this.close(id);
        }, duration);

        // 存储定时器ID
        notification.dataset.timer = timer;

        return id;
    }

    createNotification(id, message, type) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.dataset.id = id;

        const icon = this.getIconForType(type);
        
        notification.innerHTML = `
            <div class="notification-icon">${icon}</div>
            <div class="notification-content">${message}</div>
            <button class="notification-close" aria-label="关闭">×</button>
            <div class="notification-progress">
                <div class="notification-progress-bar"></div>
            </div>
        `;

        // 添加关闭按钮事件
        const closeButton = notification.querySelector('.notification-close');
        closeButton.addEventListener('click', () => this.close(id));

        // 添加鼠标悬停暂停
        notification.addEventListener('mouseenter', () => {
            const timer = notification.dataset.timer;
            if (timer) {
                clearTimeout(timer);
                const progressBar = notification.querySelector('.notification-progress-bar');
                progressBar.style.transition = 'none';
            }
        });

        // 鼠标离开继续倒计时
        notification.addEventListener('mouseleave', () => {
            const progressBar = notification.querySelector('.notification-progress-bar');
            const remainingWidth = parseFloat(getComputedStyle(progressBar).width) / progressBar.parentElement.offsetWidth * 100;
            const remainingTime = (remainingWidth / 100) * 5000;
            
            progressBar.style.transition = `width ${remainingTime}ms linear`;
            progressBar.style.width = '0%';
            
            const timer = setTimeout(() => {
                this.close(id);
            }, remainingTime);
            notification.dataset.timer = timer;
        });

        return notification;
    }

    getIconForType(type) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        return icons[type] || icons.info;
    }

    close(id) {
        const notification = this.notifications.get(id);
        if (!notification) return;

        notification.classList.remove('show');
        
        // 清除定时器
        const timer = notification.dataset.timer;
        if (timer) {
            clearTimeout(timer);
        }

        // 移除元素
        setTimeout(() => {
            if (notification.parentElement) {
                notification.parentElement.removeChild(notification);
            }
            this.notifications.delete(id);
        }, 300);
    }
}

// 创建全局通知系统实例
const notificationSystem = new NotificationSystem();

// 替换原有的showToast函数
function showToast(message, type = 'info') {
    notificationSystem.show(message, type);
}

// 工具函数：生成随机数据
function generateRandomData(count, min, max) {
    return Array.from({length: count}, () => 
        Math.floor(Math.random() * (max - min + 1)) + min
    );
}

// 导出报告功能
async function exportReport() {
    try {
        const reportContent = generateReportContent();
        const element = document.createElement('div');
        element.innerHTML = reportContent;
        document.body.appendChild(element);

        const opt = {
            margin: 10,
            filename: `energy_report_${new Date().getTime()}.pdf`,
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { scale: 2 },
            jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
        };

        await html2pdf().set(opt).from(element).save();
        document.body.removeChild(element);
        showToast('报告导出成功', 'success');
    } catch (error) {
        console.error('报告导出失败:', error);
        showToast('报告导出失败', 'error');
    }
}

// 生成报告内容
function generateReportContent() {
    const now = new Date();
    const reportDate = now.toLocaleString('zh-CN');
    
    return `
        <div class="report-content">
            <h1>能源预测分析报告</h1>
            <p>生成时间: ${reportDate}</p>
            
            <h2>1. 发电预测分析</h2>
            <p>当前发电量: 256.8 MW</p>
            <p>预测准确率: 95.2%</p>
            
            <h2>2. 负荷预测分析</h2>
            <p>当前负荷: 189.2 MW</p>
            <p>预测准确率: 93.8%</p>
            
            <h2>3. 系统状态</h2>
            <p>CPU使用率: 45%</p>
            <p>内存使用率: 62%</p>
            
            <h2>4. 告警信息</h2>
            <p>未处理告警: 3条</p>
            <p>已处理告警: 12条</p>
        </div>
    `;
}

// 修改 initChartWithLoading 函数
async function initChartWithLoading(chartId, initFunction) {
    try {
        return await initFunction();
    } catch (error) {
        console.error(`Failed to initialize chart ${chartId}:`, error);
    }
}

// 修改 refreshData 函数
async function refreshData() {
    try {
        await updateChartData();
    } catch (error) {
        console.error('Failed to refresh data:', error);
    }
}

// 时间范围标签生成函数
function generateTimeLabels(range) {
    switch(range) {
        case '24h':
            return ['06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00'];
        case '7d':
            return Array.from({length: 7}, (_, i) => `Day ${i + 1}`);
        case '30d':
            return Array.from({length: 30}, (_, i) => `Day ${i + 1}`);
        default:
            return ['06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00'];
    }
}

// 生成更真实的时间序列数据
function generateTimeSeriesData(range, baseValue, amplitude, pattern = 'default') {
    let count;
    switch(range) {
        case '24h': count = 8; break;
        case '7d': count = 7; break;
        case '30d': count = 30; break;
        default: count = 8;
    }

    const data = [];
    for (let i = 0; i < count; i++) {
        let value;
        
        switch(pattern) {
            case 'solar':
                // 光伏发电模式：日出日落曲线
                if (range === '24h') {
                    const hour = (i * 3) % 24;
                    const dayEffect = Math.sin((hour - 6) * Math.PI / 12);
                    value = baseValue + amplitude * dayEffect;
                } else {
                    // 多天数据考虑天气影响
                    const weatherEffect = Math.random() * 0.3 + 0.7; // 70%-100%效率
                    value = baseValue * weatherEffect;
                }
                break;

            case 'wind':
                // 风电模式：较大波动
                if (range === '24h') {
                    value = baseValue + (Math.random() - 0.5) * amplitude;
                } else {
                    // 多天数据考虑季节性
                    const seasonalEffect = Math.sin(i * Math.PI / count);
                    value = baseValue + amplitude * seasonalEffect;
                }
                break;

            case 'load':
                // 负荷模式：工作日规律
                if (range === '24h') {
                    const hour = (i * 3) % 24;
                    const workdayEffect = hour >= 8 && hour <= 18 ? 1.2 : 0.8;
                    value = baseValue * workdayEffect;
                } else {
                    // 多天数据考虑工作日/周末
                    const isWeekend = i % 7 >= 5;
                    value = baseValue * (isWeekend ? 0.8 : 1.2);
                }
                break;

            default:
                // 默认模式：正弦波动
                const trend = (i / count) * amplitude * 0.5;
                const variation = Math.sin(i * Math.PI / (count/2)) * amplitude;
                value = baseValue + variation + trend;
        }

        // 添加随机波动并确保值在合理范围内
        const noise = (Math.random() - 0.5) * amplitude * 0.2;
        value = Math.max(0, Math.min(baseValue * 2, value + noise));
        data.push(Math.round(value * 10) / 10); // 保留一位小数
    }

    return data;
}

// 更新图表数据函数
async function updateChartData(range = '24h') {
    try {
        const timeLabels = generateTimeLabels(range);
        
        const mockData = {
            power: {
                solar: generateTimeSeriesData(range, 150, 100, 'solar'),
                wind: generateTimeSeriesData(range, 120, 80, 'wind')
            },
            load: generateTimeSeriesData(range, 200, 100, 'load'),
            price: generateTimeSeriesData(range, 0.65, 0.2),
            charging: generateTimeSeriesData(range, 60, 40)
        };

        const charts = {
            powerPredictionChart: Chart.getChart('powerPredictionChart'),
            loadPredictionChart: Chart.getChart('loadPredictionChart'),
            pricePredictionChart: Chart.getChart('pricePredictionChart'),
            chargingPredictionChart: Chart.getChart('chargingPredictionChart')
        };

        // 更新所有图表
        await Promise.all(Object.entries(charts).map(async ([chartId, chart]) => {
            if (chart) {
                // 更新标签
                chart.data.labels = timeLabels;
                
                // 更新数据
                if (chartId === 'powerPredictionChart') {
                    chart.data.datasets[0].data = mockData.power.solar;
                    chart.data.datasets[1].data = mockData.power.wind;
                } else {
                    const dataKey = chartId.replace('PredictionChart', '');
                    chart.data.datasets[0].data = mockData[dataKey];
                }

                // 调整Y轴范围
                if (range !== '24h') {
                    const maxValue = Math.max(...chart.data.datasets.flatMap(d => d.data));
                    chart.options.scales.y.max = Math.ceil(maxValue * 1.2 / 100) * 100;
                }

                await chart.update('none');
            }
        }));

        // 更新指标数据
        updateMetricsForTimeRange(mockData, range);

    } catch (error) {
        console.error('Error updating chart data:', error);
        throw error;
    }
}

// 更新指标数据
function updateMetricsForTimeRange(data, range) {
    const getLatestValue = (data) => {
        return data[data.length - 1];
    };

    const getPeakValue = (data) => {
        return Math.max(...data);
    };

    const calculateAccuracy = () => {
        return (90 + Math.random() * 5).toFixed(1); // 90-95%的准确率
    };

    // 更新发电预测指标
    const powerMetrics = document.querySelector('#powerMetrics');
    if (powerMetrics) {
        const totalPower = getLatestValue(data.power.solar) + getLatestValue(data.power.wind);
        const peakPower = Math.max(getPeakValue(data.power.solar) + getPeakValue(data.power.wind));
        
        powerMetrics.querySelector('.current-value').textContent = `${totalPower.toFixed(1)} MW`;
        powerMetrics.querySelector('.peak-value').textContent = `${peakPower.toFixed(1)} MW`;
        powerMetrics.querySelector('.accuracy-value').textContent = `${calculateAccuracy()}%`;
    }

    // 更新其他指标...（负荷、电价、充电等）
    // ... 类似的更新逻辑
}

// 添加图表交互增强功能
function enhanceChartInteractions() {
    const charts = {
        power: Chart.getChart('powerPredictionChart'),
        load: Chart.getChart('loadPredictionChart'),
        price: Chart.getChart('pricePredictionChart'),
        charging: Chart.getChart('chargingPredictionChart')
    };

    // 为每个图表添加点击事件处理
    Object.entries(charts).forEach(([type, chart]) => {
        if (!chart) return;

        chart.options.onClick = (event, elements) => {
            if (elements.length > 0) {
                const dataIndex = elements[0].index;
                showDetailPopup(type, dataIndex, chart.data.datasets[0].data[dataIndex]);
            }
        };

        // 添加悬停效果增强
        chart.options.plugins.tooltip = {
            ...chart.options.plugins.tooltip,
            callbacks: {
                label: function(context) {
                    let label = context.dataset.label || '';
                    if (label) {
                        label += ': ';
                    }
                    const value = context.parsed.y;
                    const unit = getUnitByChartType(type);
                    return `${label}${value.toFixed(2)} ${unit}`;
                }
            }
        };

        // 更新图表以应用新的配置
        chart.update();
    });
}

// 显示详细信息弹窗
function showDetailPopup(type, index, value) {
    const titles = {
        power: '发电预测详情',
        load: '负荷预测详情',
        price: '电价预测详情',
        charging: '充电预测详情'
    };

    const units = {
        power: 'MW',
        load: 'MW',
        price: '¥/kWh',
        charging: 'kW'
    };

    const timeLabels = ['06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00'];

    // 创建弹窗内容
    const content = `
        <div class="detail-popup">
            <h5 class="mb-3">${titles[type]}</h5>
            <div class="detail-info">
                <p><strong>时间点：</strong>${timeLabels[index]}</p>
                <p><strong>预测值：</strong>${value.toFixed(2)} ${units[type]}</p>
                <p><strong>置信区间：</strong>${(value * 0.95).toFixed(2)} - ${(value * 1.05).toFixed(2)} ${units[type]}</p>
                <p><strong>预测准确率：</strong>${(95 - Math.random() * 5).toFixed(1)}%</p>
            </div>
            <div class="detail-actions mt-3">
                <button class="btn btn-sm btn-primary" onclick="exportDetailReport('${type}', ${index})">
                    导出详情
                </button>
                <button class="btn btn-sm btn-info" onclick="showHistoricalComparison('${type}', ${index})">
                    历史对比
                </button>
            </div>
        </div>
    `;

    // 使用Bootstrap的Modal显示弹窗
    const modal = new bootstrap.Modal(document.createElement('div'));
    modal.element.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">${titles[type]}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modal.element);
    modal.show();

    // 自动清理
    modal.element.addEventListener('hidden.bs.modal', () => {
        document.body.removeChild(modal.element);
    });
}

// 显示历史对比数据
function showHistoricalComparison(type, index) {
    const timeLabels = ['06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00'];
    const currentTime = timeLabels[index];
    
    // 创建历史对比图表
    const modal = new bootstrap.Modal(document.createElement('div'));
    modal.element.innerHTML = `
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">历史数据对比 - ${currentTime}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <canvas id="historicalComparisonChart"></canvas>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modal.element);
    modal.show();

    // 初始化对比图表
    const ctx = document.getElementById('historicalComparisonChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 7}, (_, i) => `Day ${i + 1}`),
            datasets: [{
                label: '预测值',
                data: generateRandomData(7, 80, 120),
                borderColor: '#4F8CBE',
                tension: 0.4
            }, {
                label: '实际值',
                data: generateRandomData(7, 80, 120),
                borderColor: '#20c997',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: `近7天${currentTime}时段数据对比`
                }
            }
        }
    });

    // 自动清理
    modal.element.addEventListener('hidden.bs.modal', () => {
        document.body.removeChild(modal.element);
    });
}

// 获取图表类型对应的单位
function getUnitByChartType(type) {
    const units = {
        power: 'MW',
        load: 'MW',
        price: '¥/kWh',
        charging: 'kW'
    };
    return units[type] || '';
}

// 生成详细报告内容
async function generateDetailReport(type, index) {
    const timeLabels = ['06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00'];
    const currentTime = timeLabels[index];
    const chart = Chart.getChart(`${type}PredictionChart`);
    const value = chart.data.datasets[0].data[index];
    
    return `
        <div class="report-content">
            <h1>${type}预测详细报告</h1>
            <p>生成时间：${new Date().toLocaleString('zh-CN')}</p>
            <p>预测时间点：${currentTime}</p>
            
            <h2>预测详情</h2>
            <table>
                <tr>
                    <th>预测值</th>
                    <td>${value.toFixed(2)} ${getUnitByChartType(type)}</td>
                </tr>
                <tr>
                    <th>置信区间</th>
                    <td>${(value * 0.95).toFixed(2)} - ${(value * 1.05).toFixed(2)} ${getUnitByChartType(type)}</td>
                </tr>
                <tr>
                    <th>预测准确率</th>
                    <td>${(95 - Math.random() * 5).toFixed(1)}%</td>
                </tr>
            </table>
            
            <h2>影响因素分析</h2>
            <ul>
                <li>天气状况：晴朗</li>
                <li>温度：25°C</li>
                <li>历史同期数据偏差：-2.3%</li>
            </ul>
            
            <h2>建议措施</h2>
            <ol>
                <li>密切监控系统运行状态</li>
                <li>做好应急预案准备</li>
                <li>优化调度策略</li>
            </ol>
        </div>
    `;
}

// 添加图表切换功能
function initChartSwitching() {
    const chartTypeButtons = document.querySelectorAll('#chartTypeToggle .btn');
    const chartContainers = document.querySelectorAll('.chart-container');
    
    chartTypeButtons.forEach(button => {
        button.addEventListener('click', function() {
            const chartType = this.dataset.chartType;
            
            // 更新按钮状态
            chartTypeButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            // 更新图表显示
            chartContainers.forEach(container => {
                if (container.id === `${chartType}ChartContainer`) {
                    container.classList.add('active');
                    // 确保图表重绘
                    const chart = Chart.getChart(container.querySelector('canvas').id);
                    if (chart) {
                        setTimeout(() => {
                            chart.resize();
                            chart.update('none'); // 使用 'none' 模式避免动画
                        }, 300); // 等待过渡动画完成
                    }
                } else {
                    container.classList.remove('active');
                }
            });
            
            // 更新指标显示
            document.querySelectorAll('.metrics-group').forEach(group => {
                group.classList.remove('active');
            });
            document.getElementById(`${chartType}Metrics`).classList.add('active');
        });
    });
}

// 初始化时间范围控制
function initTimeRangeControl() {
    const timeRangeButtons = document.querySelectorAll('#timeRangeToggle .btn');
    
    timeRangeButtons.forEach(button => {
        button.addEventListener('click', async function() {
            const range = this.dataset.range;
            
            // 更新按钮状态
            timeRangeButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            // 显示加载状态
            const chartTypes = ['power', 'load', 'price', 'charging'];
            chartTypes.forEach(type => {
                showChartLoading(`${type}PredictionChart`);
            });
            
            try {
                await updateChartData(range);
                notificationSystem.show('数据更新成功', 'success');
            } catch (error) {
                console.error('Error updating time range:', error);
                notificationSystem.show('数据加载失败，请重试', 'error');
            } finally {
                chartTypes.forEach(type => {
                    hideChartLoading(`${type}PredictionChart`);
                });
            }
        });
    });
}

// 更新图表显示选项
function initChartOptions() {
    // 置信区间控制
    document.getElementById('showConfidenceInterval').addEventListener('change', function() {
        const charts = getAllCharts();
        const show = this.checked;
        
        Object.values(charts).forEach(chart => {
            if (!chart) return;
            
            // 更新置信区间数据集
            const mainData = chart.data.datasets[0].data;
            if (show) {
                // 添加置信区间
                chart.data.datasets.push({
                    label: '置信区间',
                    data: mainData.map(v => [v * 0.95, v * 1.05]),
                    type: 'line',
                    fill: true,
                    backgroundColor: 'rgba(200, 200, 200, 0.2)',
                    borderWidth: 0
                });
            } else {
                // 移除置信区间
                chart.data.datasets = chart.data.datasets.filter(ds => ds.label !== '置信区间');
            }
            
            chart.update();
        });
    });
    
    // 趋势线控制
    document.getElementById('showTrend').addEventListener('change', function() {
        const charts = getAllCharts();
        const show = this.checked;
        
        Object.values(charts).forEach(chart => {
            if (!chart) return;
            
            // 更新趋势线数据集
            const mainData = chart.data.datasets[0].data;
            if (show) {
                // 添加趋势线
                const trendData = calculateTrendLine(mainData);
                chart.data.datasets.push({
                    label: '趋势线',
                    data: trendData,
                    type: 'line',
                    borderColor: 'rgba(255, 99, 132, 0.8)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false
                });
            } else {
                // 移除趋势线
                chart.data.datasets = chart.data.datasets.filter(ds => ds.label !== '趋势线');
            }
            
            chart.update();
        });
    });
}

// 计算趋势线数据
function calculateTrendLine(data) {
    const n = data.length;
    let sumX = 0;
    let sumY = 0;
    let sumXY = 0;
    let sumXX = 0;
    
    for (let i = 0; i < n; i++) {
        sumX += i;
        sumY += data[i];
        sumXY += i * data[i];
        sumXX += i * i;
    }
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    return data.map((_, i) => slope * i + intercept);
}

// 获取所有图表实例
function getAllCharts() {
    return {
        power: Chart.getChart('powerPredictionChart'),
        load: Chart.getChart('loadPredictionChart'),
        price: Chart.getChart('pricePredictionChart'),
        charging: Chart.getChart('chargingPredictionChart')
    };
}

// 添加分析功能
document.getElementById('analyzeBtn').addEventListener('click', function() {
    const currentChartType = document.querySelector('#chartTypeToggle .btn.active').dataset.chartType;
    const chart = Chart.getChart(`${currentChartType}PredictionChart`);
    
    if (!chart) return;
    
    const data = chart.data.datasets[0].data;
    const analysis = analyzeData(data);
    
    showAnalysisModal(currentChartType, analysis);
});

// 数据分析函数
function analyzeData(data) {
    const mean = data.reduce((a, b) => a + b) / data.length;
    const variance = data.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / data.length;
    const stdDev = Math.sqrt(variance);
    const max = Math.max(...data);
    const min = Math.min(...data);
    
    // 计算变化率
    const changes = [];
    for (let i = 1; i < data.length; i++) {
        changes.push((data[i] - data[i-1]) / data[i-1] * 100);
    }
    const avgChange = changes.reduce((a, b) => a + b) / changes.length;
    
    return {
        mean: mean.toFixed(2),
        stdDev: stdDev.toFixed(2),
        max: max.toFixed(2),
        min: min.toFixed(2),
        avgChange: avgChange.toFixed(2),
        volatility: (stdDev / mean * 100).toFixed(2)
    };
}

// 显示分析结果弹窗
function showAnalysisModal(type, analysis) {
    const titles = {
        power: '发电数据分析',
        load: '负荷数据分析',
        price: '电价数据分析',
        charging: '充电数据分析'
    };
    
    const units = {
        power: 'MW',
        load: 'MW',
        price: '¥/kWh',
        charging: 'kW'
    };
    
    const modal = new bootstrap.Modal(document.createElement('div'));
    modal.element.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">${titles[type]}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="table-responsive">
                        <table class="table">
                            <tbody>
                                <tr>
                                    <th>平均值</th>
                                    <td>${analysis.mean} ${units[type]}</td>
                                </tr>
                                <tr>
                                    <th>标准差</th>
                                    <td>${analysis.stdDev} ${units[type]}</td>
                                </tr>
                                <tr>
                                    <th>最大值</th>
                                    <td>${analysis.max} ${units[type]}</td>
                                </tr>
                                <tr>
                                    <th>最小值</th>
                                    <td>${analysis.min} ${units[type]}</td>
                                </tr>
                                <tr>
                                    <th>平均变化率</th>
                                    <td>${analysis.avgChange}%</td>
                                </tr>
                                <tr>
                                    <th>波动率</th>
                                    <td>${analysis.volatility}%</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    <div class="analysis-summary mt-3">
                        <h6>分析结论</h6>
                        <p>${generateAnalysisSummary(type, analysis)}</p>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal.element);
    modal.show();
    
    modal.element.addEventListener('hidden.bs.modal', () => {
        document.body.removeChild(modal.element);
    });
}

// 生成分析总结
function generateAnalysisSummary(type, analysis) {
    const summaries = {
        power: `发电量整体表现${analysis.avgChange > 0 ? '上升' : '下降'}趋势，波动率为${analysis.volatility}%。`,
        load: `负荷需求波动率为${analysis.volatility}%，平均变化率${Math.abs(analysis.avgChange)}%。`,
        price: `电价走势${analysis.avgChange > 0 ? '上涨' : '下跌'}，波动幅度${analysis.volatility}%。`,
        charging: `充电需求变化率${Math.abs(analysis.avgChange)}%，整体波动率${analysis.volatility}%。`
    };
    
    return summaries[type] + ` 建议持续监控，做好${analysis.volatility > 10 ? '波动' : '稳定'}预期。`;
}

// 添加图表容器清理函数
function cleanupChartContainer(chartId) {
    const chart = Chart.getChart(chartId);
    if (chart) {
        chart.destroy();
    }
}

// 移动端导航交互
document.addEventListener('DOMContentLoaded', function() {
    // 侧边菜单控制
    const menuToggle = document.getElementById('menuToggle');
    const sideMenu = document.getElementById('sideMenu');
    const overlay = document.getElementById('overlay');
    
    function toggleMenu() {
        sideMenu.classList.toggle('active');
        overlay.classList.toggle('active');
        document.body.style.overflow = sideMenu.classList.contains('active') ? 'hidden' : '';
    }
    
    menuToggle.addEventListener('click', toggleMenu);
    overlay.addEventListener('click', toggleMenu);
    
    // 处理触摸滑动手势
    let touchStartX = 0;
    let touchEndX = 0;
    
    document.addEventListener('touchstart', e => {
        touchStartX = e.changedTouches[0].screenX;
    }, false);
    
    document.addEventListener('touchend', e => {
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    }, false);
    
    function handleSwipe() {
        const swipeDistance = touchEndX - touchStartX;
        const threshold = 100; // 最小滑动距离
        
        if (Math.abs(swipeDistance) < threshold) return;
        
        if (swipeDistance > 0 && touchStartX < 30) {
            // 从左向右滑动，打开菜单
            sideMenu.classList.add('active');
            overlay.classList.add('active');
            document.body.style.overflow = 'hidden';
        } else if (swipeDistance < 0 && sideMenu.classList.contains('active')) {
            // 从右向左滑动，关闭菜单
            sideMenu.classList.remove('active');
            overlay.classList.remove('active');
            document.body.style.overflow = '';
        }
    }
    
    // 底部导航切换
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            navItems.forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');
        });
    });
    
    // 二级导航切换
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const range = this.dataset.tab;
            tabBtns.forEach(tab => tab.classList.remove('active'));
            this.classList.add('active');
            updateChartData(range);
        });
    });
    
    // 通知按钮点击事件
    const notificationBtn = document.getElementById('notificationBtn');
    if (notificationBtn) {
        notificationBtn.addEventListener('click', function() {
            // 这里可以添加显示通知列表的逻辑
            notificationSystem.show('正在加载通知...', 'info');
        });
    }
}); 