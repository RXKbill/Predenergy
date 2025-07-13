// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    configManagement.init();
    });

// 确保在DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing system configuration management...');
    
    // 验证SweetAlert2是否可用
    if (typeof Swal === 'undefined') {
        console.error('SweetAlert2 is not loaded!');
    } else {
        console.log('SweetAlert2 is loaded successfully');
    }
    
    // 初始化配置管理
    configManagement.init();
});


 // 系统配置管理对象
 const configManagement = {
    // 当前配置版本
    currentVersion: 'v2.1.3',
    
    // 配置历史记录
    configHistory: [
        {
            version: 'v2.1.3',
            date: '2024-03-21 15:30:00',
            operator: '管理员',
            description: '优化预测模型参数',
            config: {
                thresholds: {
                    windThreshold: 15,
                    loadThreshold: 10,
                    confidenceThreshold: 85
                },
                modelParams: {
                    windowSize: 168,
                    attentionHeads: 8,
                    learningRate: 0.001
                },
                alertSettings: {
                    level: 'warning',
                    methods: {
                        email: true,
                        sms: true,
                        system: true
                    },
                    time: 'always'
                }
            }
        },
        {
            version: 'v2.1.2',
            date: '2024-03-20 14:20:00',
            operator: '管理员',
            description: '调整告警阈值',
            config: {
                thresholds: {
                    windThreshold: 20,
                    loadThreshold: 10,
                    confidenceThreshold: 80
                },
                modelParams: {
                    windowSize: 168,
                    attentionHeads: 8,
                    learningRate: 0.001
                },
                alertSettings: {
                    level: 'warning',
                    methods: {
                        email: true,
                        sms: true,
                        system: true
                    },
                    time: 'always'
                }
            }
        }
    ],

    // 查看历史版本
    viewConfigHistory() {
        Swal.fire({
            title: '配置历史',
            html: `
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>版本</th>
                                <th>时间</th>
                                <th>操作人</th>
                                <th>说明</th>
                                <th>操作</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${this.configHistory.map(history => `
                                <tr>
                                    <td>${history.version}</td>
                                    <td>${history.date}</td>
                                    <td>${history.operator}</td>
                                    <td>${history.description}</td>
                                    <td>
                                        <div class="btn-group">
                                            <button class="btn btn-sm btn-outline-primary" onclick="window.viewHistoryDetail('${history.version}')">
                                                <i class="fas fa-eye"></i>
                                            </button>
                                            <button class="btn btn-sm btn-outline-warning" onclick="window.rollbackConfig('${history.version}')">
                                                <i class="fas fa-history"></i> 回溯
                                            </button>
                                        </div>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `,
            width: '800px',
            showCloseButton: true,
            showConfirmButton: false
        });
    },

    // 版本对比
    compareVersions() {
        // 获取可比较的版本列表
        const versions = this.configHistory.map(h => ({
            version: h.version,
            date: h.date
        }));

        Swal.fire({
            title: '版本对比',
            html: `
                <div class="row mb-3">
                    <div class="col-6">
                        <label class="form-label">基准版本</label>
                        <select class="form-select" id="baseVersion">
                            ${versions.map(v => `
                                <option value="${v.version}"${v.version === this.currentVersion ? ' selected' : ''}>
                                    ${v.version} (${v.version === this.currentVersion ? '当前' : v.date})
                                </option>
                            `).join('')}
                        </select>
                    </div>
                    <div class="col-6">
                        <label class="form-label">对比版本</label>
                        <select class="form-select" id="compareVersion">
                            ${versions.map(v => `
                                <option value="${v.version}"${v.version === versions[1]?.version ? ' selected' : ''}>
                                    ${v.version} (${v.version === this.currentVersion ? '当前' : v.date})
                                </option>
                            `).join('')}
                        </select>
                    </div>
                </div>
                <div class="table-responsive mt-3">
                    <table class="table table-hover table-bordered" id="diffTable">
                        <thead>
                            <tr>
                                <th>参数</th>
                                <th>基准版本值</th>
                                <th>对比版本值</th>
                                <th>差异</th>
                            </tr>
                        </thead>
                        <tbody>
                            <!-- 差异内容将通过JavaScript动态更新 -->
                        </tbody>
                    </table>
                </div>
            `,
            width: '800px',
            showConfirmButton: false,
            showCloseButton: true,
            didOpen: () => {
                // 初始化显示差异
                this.updateVersionDiff();
                
                // 监听版本选择变化
                document.getElementById('baseVersion').addEventListener('change', () => this.updateVersionDiff());
                document.getElementById('compareVersion').addEventListener('change', () => this.updateVersionDiff());
            }
        });
    },

    // 更新版本差异显示
    updateVersionDiff() {
        const baseVersion = document.getElementById('baseVersion').value;
        const compareVersion = document.getElementById('compareVersion').value;

        const baseConfig = this.configHistory.find(h => h.version === baseVersion)?.config;
        const compareConfig = this.configHistory.find(h => h.version === compareVersion)?.config;

        if (!baseConfig || !compareConfig) return;

        const diffTable = document.getElementById('diffTable').getElementsByTagName('tbody')[0];
        diffTable.innerHTML = '';

        // 比较阈值
        this.addDiffRow(diffTable, '风电产能波动阈值', baseConfig.thresholds.windThreshold, compareConfig.thresholds.windThreshold, '%');
        this.addDiffRow(diffTable, '负荷预测偏差阈值', baseConfig.thresholds.loadThreshold, compareConfig.thresholds.loadThreshold, '%');
        this.addDiffRow(diffTable, '时序预测置信度阈值', baseConfig.thresholds.confidenceThreshold, compareConfig.thresholds.confidenceThreshold, '%');

        // 比较模型参数
        this.addDiffRow(diffTable, '时序窗口大小', baseConfig.modelParams.windowSize, compareConfig.modelParams.windowSize, '小时');
        this.addDiffRow(diffTable, '注意力头数', baseConfig.modelParams.attentionHeads, compareConfig.modelParams.attentionHeads, '个');
        this.addDiffRow(diffTable, '学习率', baseConfig.modelParams.learningRate, compareConfig.modelParams.learningRate);
    },

    // 添加差异行
    addDiffRow(table, paramName, baseValue, compareValue, unit = '') {
        const row = table.insertRow();
        row.insertCell(0).textContent = paramName;
        row.insertCell(1).textContent = baseValue + unit;
        row.insertCell(2).textContent = compareValue + unit;
        
        const diff = compareValue - baseValue;
        const diffCell = row.insertCell(3);
        if (diff === 0) {
            diffCell.innerHTML = '<span class="text-muted">无变化</span>';
                } else {
            const sign = diff > 0 ? '+' : '';
            diffCell.innerHTML = `<span class="text-${diff > 0 ? 'success' : 'danger'}">${sign}${diff}${unit}</span>`;
        }
    },

    // 同步配置
    syncConfig() {
        Swal.fire({
            title: '同步配置',
            html: `
                <div class="mb-4">
                    <div class="alert alert-info">
                        <i class="bx bx-info-circle me-2"></i>
                        选择要同步的目标环境和配置项
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-6">
                        <h6 class="mb-3">目标环境</h6>
                        <div class="form-check mb-2">
                            <input class="form-check-input" type="checkbox" id="syncProd" checked>
                            <label class="form-check-label" for="syncProd">
                                生产环境
                            </label>
                        </div>
                        <div class="form-check mb-2">
                            <input class="form-check-input" type="checkbox" id="syncTest">
                            <label class="form-check-label" for="syncTest">
                                测试环境
                            </label>
                        </div>
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="syncDev">
                            <label class="form-check-label" for="syncDev">
                                开发环境
                            </label>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <h6 class="mb-3">配置项</h6>
                        <div class="form-check mb-2">
                            <input class="form-check-input" type="checkbox" id="syncThresholds" checked>
                            <label class="form-check-label" for="syncThresholds">
                                告警阈值
                            </label>
                        </div>
                        <div class="form-check mb-2">
                            <input class="form-check-input" type="checkbox" id="syncModel" checked>
                            <label class="form-check-label" for="syncModel">
                                模型参数
                            </label>
                        </div>
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="syncAlert" checked>
                            <label class="form-check-label" for="syncAlert">
                                告警设置
                            </label>
                        </div>
                    </div>
                </div>
            `,
            showCancelButton: true,
            confirmButtonText: '开始同步',
            cancelButtonText: '取消',
            preConfirm: () => {
                const environments = {
                    prod: document.getElementById('syncProd').checked,
                    test: document.getElementById('syncTest').checked,
                    dev: document.getElementById('syncDev').checked
                };
                const items = {
                    thresholds: document.getElementById('syncThresholds').checked,
                    model: document.getElementById('syncModel').checked,
                    alert: document.getElementById('syncAlert').checked
                };

                if (!Object.values(environments).some(v => v)) {
                    Swal.showValidationMessage('请至少选择一个目标环境');
                    return false;
                }
                if (!Object.values(items).some(v => v)) {
                    Swal.showValidationMessage('请至少选择一个配置项');
                    return false;
                }

                return { environments, items };
            }
        }).then((result) => {
            if (result.isConfirmed) {
                // 模拟同步过程
                Swal.fire({
                    title: '正在同步',
                    html: `
                        <div class="progress">
                            <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%"></div>
                        </div>
                        <div id="syncStatus" class="mt-3">准备同步...</div>
                    `,
                    showConfirmButton: false,
                    allowOutsideClick: false,
                    didOpen: () => {
                        const progressBar = Swal.getHtmlContainer().querySelector('.progress-bar');
                        const statusText = Swal.getHtmlContainer().querySelector('#syncStatus');
                        let progress = 0;

                        const interval = setInterval(() => {
                            progress += 5;
                            progressBar.style.width = `${progress}%`;
                            
                            if (progress === 30) {
                                statusText.textContent = '正在同步告警阈值...';
                            } else if (progress === 60) {
                                statusText.textContent = '正在同步模型参数...';
                            } else if (progress === 90) {
                                statusText.textContent = '正在同步告警设置...';
                            }

                            if (progress >= 100) {
                                clearInterval(interval);
                                setTimeout(() => {
                                    Swal.fire({
                                        title: '同步完成',
                                        text: '配置已成功同步到选定环境',
                                        icon: 'success'
                                    });
                                }, 500);
                            }
                        }, 100);
                    }
                });
            }
        });
    },

    // 查看配置详情
    viewConfigDetail() {
        const currentConfig = {
            thresholds: {
                windThreshold: parseFloat(document.getElementById('windInput').value),
                loadThreshold: parseFloat(document.getElementById('loadInput').value),
                confidenceThreshold: parseFloat(document.getElementById('confidenceInput').value)
            },
            modelParams: {
                windowSize: parseInt(document.getElementById('windowSizeInput').value),
                attentionHeads: parseInt(document.getElementById('attentionHeadsInput').value),
                learningRate: parseFloat(document.getElementById('learningRateInput').value)
            },
            alertSettings: {
                level: document.getElementById('alertLevel').value,
                methods: {
                    email: document.getElementById('alertEmail').checked,
                    sms: document.getElementById('alertSMS').checked,
                    system: document.getElementById('alertSystem').checked
                },
                time: document.getElementById('alertTime').value
            }
        };

        Swal.fire({
            title: '配置详情',
            html: `
                <div class="accordion" id="configAccordion">
                    <div class="accordion-item">
                        <h2 class="accordion-header">
                            <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#thresholdConfig">
                                阈值配置
                            </button>
                        </h2>
                        <div id="thresholdConfig" class="accordion-collapse collapse show" data-bs-parent="#configAccordion">
                            <div class="accordion-body">
                                <table class="table table-sm">
                                    <tr>
                                        <td>风电产能波动阈值</td>
                                        <td>${currentConfig.thresholds.windThreshold}%</td>
                                    </tr>
                                    <tr>
                                        <td>负荷预测偏差阈值</td>
                                        <td>${currentConfig.thresholds.loadThreshold}%</td>
                                    </tr>
                                    <tr>
                                        <td>时序预测置信度阈值</td>
                                        <td>${currentConfig.thresholds.confidenceThreshold}%</td>
                                    </tr>
                                </table>
                            </div>
                        </div>
                    </div>
                    <div class="accordion-item">
                        <h2 class="accordion-header">
                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#modelConfig">
                                模型参数
                            </button>
                        </h2>
                        <div id="modelConfig" class="accordion-collapse collapse" data-bs-parent="#configAccordion">
                            <div class="accordion-body">
                                <table class="table table-sm">
                                    <tr>
                                        <td>时序窗口大小</td>
                                        <td>${currentConfig.modelParams.windowSize}小时</td>
                                    </tr>
                                    <tr>
                                        <td>注意力头数</td>
                                        <td>${currentConfig.modelParams.attentionHeads}个</td>
                                    </tr>
                                    <tr>
                                        <td>学习率</td>
                                        <td>${currentConfig.modelParams.learningRate}</td>
                                    </tr>
                                </table>
                            </div>
                        </div>
                    </div>
                    <div class="accordion-item">
                        <h2 class="accordion-header">
                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#alertConfig">
                                告警设置
                            </button>
                        </h2>
                        <div id="alertConfig" class="accordion-collapse collapse" data-bs-parent="#configAccordion">
                            <div class="accordion-body">
                                <table class="table table-sm">
                                    <tr>
                                        <td>告警级别</td>
                                        <td>${currentConfig.alertSettings.level}</td>
                                    </tr>
                                    <tr>
                                        <td>告警方式</td>
                                        <td>
                                            ${currentConfig.alertSettings.methods.email ? '邮件通知<br>' : ''}
                                            ${currentConfig.alertSettings.methods.sms ? '短信通知<br>' : ''}
                                            ${currentConfig.alertSettings.methods.system ? '系统通知' : ''}
                                        </td>
                                    </tr>
                                    <tr>
                                        <td>告警时间</td>
                                        <td>${currentConfig.alertSettings.time}</td>
                                    </tr>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            `,
            width: '600px',
            showCloseButton: true,
            showConfirmButton: false
        });
    },

    // 初始化
    init() {
        // 添加历史版本按钮事件监听
        const historyBtn = document.getElementById('viewHistoryBtn');
        if (historyBtn) {
            historyBtn.addEventListener('click', () => this.viewConfigHistory());
        }

        // 添加版本对比按钮事件监听
        const compareBtn = document.getElementById('compareVersionBtn');
        if (compareBtn) {
            compareBtn.addEventListener('click', () => this.compareVersions());
        }

        // 添加同步配置按钮事件监听
        const syncBtn = document.getElementById('syncConfigBtn');
        if (syncBtn) {
            syncBtn.addEventListener('click', () => this.syncConfig());
        }

        // 添加配置详情按钮事件监听
        const detailBtn = document.getElementById('configDetailBtn');
        if (detailBtn) {
            detailBtn.addEventListener('click', () => this.viewConfigDetail());
        }
    }
};

document.addEventListener('DOMContentLoaded', function() {
    // 初始化工具提示
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // 初始化滑块
    initializeSliders();
    
    // 初始化表单验证
    initializeFormValidation();
});

// 初始化滑块
function initializeSliders() {
    const sliders = {
        wind: { id: 'windSlider', input: 'windInput', start: 15, min: 0, max: 30 },
        load: { id: 'loadSlider', input: 'loadInput', start: 10, min: 0, max: 20 },
        confidence: { id: 'confidenceSlider', input: 'confidenceInput', start: 85, min: 0, max: 100 }
    };

    Object.entries(sliders).forEach(([key, config]) => {
        const slider = document.getElementById(config.id);
        const input = document.getElementById(config.input);

        if (!slider || !input) return;

        noUiSlider.create(slider, {
            start: config.start,
            connect: [true, false],
            step: key === 'confidence' ? 1 : 0.1,
            range: {
                'min': config.min,
                'max': config.max
            },
            format: {
                to: value => parseFloat(value).toFixed(key === 'confidence' ? 0 : 1),
                from: value => parseFloat(value)
            }
        });

        // 滑块值变化时更新输入框
                slider.noUiSlider.on('update', values => {
            input.value = values[0];
        });

        // 输入框值变化时更新滑块
        input.addEventListener('change', function() {
            let value = parseFloat(this.value);
            if (isNaN(value)) value = config.start;
            if (value < config.min) value = config.min;
            if (value > config.max) value = config.max;
            slider.noUiSlider.set(value);
        });
    });
}

// 表单验证初始化
function initializeFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
}

// 配置版本管理
function showVersionHistory() {
    Swal.fire({
        title: '配置版本历史',
        html: `
            <div class="table-responsive">
                <table class="table table-bordered">
                    <thead>
                        <tr>
                            <th>版本号</th>
                            <th>更新时间</th>
                            <th>操作人</th>
                            <th>操作</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>v2.1.3</td>
                            <td>2024-03-21 15:30</td>
                            <td>管理员</td>
                            <td><button class="btn btn-sm btn-primary" onclick="restoreVersion('v2.1.3')">恢复</button></td>
                        </tr>
                        <tr>
                            <td>v2.1.2</td>
                            <td>2024-03-20 14:20</td>
                            <td>管理员</td>
                            <td><button class="btn btn-sm btn-primary" onclick="restoreVersion('v2.1.2')">恢复</button></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        `,
        width: '600px'
    });
}

// 查看操作记录详情
function viewDetail(id) {
    // 模拟操作记录数据
    const records = {
        1: {
            time: '2024-03-21 15:30',
            type: '参数更新',
            operator: '管理员',
            operatorId: 'admin001',
            department: '运维部',
            version: {
                from: 'v2.1.2',
                to: 'v2.1.3'
            },
            changes: [
                {
                    parameter: '风电产能波动阈值',
                    oldValue: '20%',
                    newValue: '15%',
                    impact: '预计可减少50%误报率',
                    riskLevel: 'low'
                },
                {
                    parameter: '时序预测置信度阈值',
                    oldValue: '80%',
                    newValue: '85%',
                    impact: '提高预测准确性',
                    riskLevel: 'medium'
                }
            ],
            description: '更新预警阈值配置',
            relatedSystems: [
                {
                    name: '风电场监控系统',
                    status: 'affected',
                    syncStatus: 'completed'
                },
                {
                    name: '负荷预测系统',
                    status: 'unaffected',
                    syncStatus: 'not_required'
                }
            ],
            audit: {
                requestId: 'REQ-2024032115301',
                approver: '技术主管',
                approvalTime: '2024-03-21 15:25',
                changeWindow: '2024-03-21 15:30 - 16:00',
                emergencyLevel: 'normal'
            },
            verification: {
                status: 'passed',
                tester: '测试工程师',
                testTime: '2024-03-21 15:45',
                testCases: '3/3 通过',
                testReport: 'TR-20240321-001'
            }
        },
        2: {
            time: '2024-03-21 14:20',
            type: '配置导入',
            operator: '管理员',
            operatorId: 'admin001',
            department: '运维部',
            version: {
                from: 'v2.1.1',
                to: 'v2.1.2'
            },
            changes: [
                {
                    parameter: '时序窗口大小',
                    oldValue: '144',
                    newValue: '168',
                    impact: '优化长期预测效果',
                    riskLevel: 'medium'
                }
            ],
            description: '导入生产环境配置',
            relatedSystems: [
                {
                    name: '预测分析系统',
                    status: 'affected',
                    syncStatus: 'completed'
                }
            ],
            audit: {
                requestId: 'REQ-2024032114201',
                approver: '技术主管',
                approvalTime: '2024-03-21 14:15',
                changeWindow: '2024-03-21 14:20 - 14:50',
                emergencyLevel: 'normal'
            },
            verification: {
                status: 'passed',
                tester: '测试工程师',
                testTime: '2024-03-21 14:35',
                testCases: '2/2 通过',
                testReport: 'TR-202403-21-002'
            }
        }
    };

    const record = records[id];
    if (!record) return;

    Swal.fire({
        title: '操作记录详情',
        html: `
            <div class="accordion" id="detailAccordion">
                <!-- 基本信息 -->
                <div class="accordion-item">
                    <h2 class="accordion-header">
                        <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#basicInfo">
                            基本信息
                        </button>
                    </h2>
                    <div id="basicInfo" class="accordion-collapse collapse show" data-bs-parent="#detailAccordion">
                        <div class="accordion-body">
                            <table class="table table-bordered">
                                <tr>
                                    <th width="30%">操作时间</th>
                                    <td>${record.time}</td>
                                </tr>
                                <tr>
                                    <th>操作类型</th>
                                    <td><span class="badge bg-${record.type === '参数更新' ? 'success' : 'warning'}">${record.type}</span></td>
                                </tr>
                                <tr>
                                    <th>操作人</th>
                                    <td>
                                        ${record.operator}
                                        <small class="text-muted">(ID: ${record.operatorId})</small>
                                        <span class="badge bg-info ms-2">${record.department}</span>
                                    </td>
                                </tr>
                                <tr>
                                    <th>版本变更</th>
                                    <td>
                                        <span class="text-muted">${record.version.from}</span>
                                        <i class="fas fa-arrow-right mx-2"></i>
                                        <span class="text-success">${record.version.to}</span>
                                    </td>
                                </tr>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        `,
        width: '800px',
        showCloseButton: true,
        showConfirmButton: false
    });
}

// 回滚配置
function rollbackConfig(version) {
    Swal.fire({
        title: '确认回滚',
        text: `是否将配置回滚到版本 ${version}？`,
        icon: 'warning',
        showCancelButton: true,
        confirmButtonText: '回滚',
        cancelButtonText: '取消'
    }).then((result) => {
        if (result.isConfirmed) {
            // 模拟回滚操作
            Swal.fire({
                title: '回滚成功',
                text: `配置已回滚到版本 ${version}`,
                icon: 'success'
            });
        }
    });
}

// 导出配置
function exportConfig() {
    const config = {
        version: CONFIG.version,
        timestamp: new Date().toISOString(),
        thresholds: {
            wind: parseFloat(document.getElementById('windInput').value),
            load: parseFloat(document.getElementById('loadInput').value),
            confidence: parseFloat(document.getElementById('confidenceInput').value)
        }
    };

    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `system_config_${config.version}.json`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

// 导入配置
function importConfig() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = e => {
        const file = e.target.files[0];
        const reader = new FileReader();
        reader.onload = event => {
            try {
                const config = JSON.parse(event.target.result);
                // 更新配置
                Object.entries(config.thresholds).forEach(([key, value]) => {
                    const slider = document.getElementById(`${key}Slider`);
                    if (slider && slider.noUiSlider) {
                        slider.noUiSlider.set(value);
                    }
                });
                
                Swal.fire({
                    title: '导入成功',
                    text: '配置已成功导入',
                    icon: 'success'
                });
            } catch (error) {
                Swal.fire({
                    title: '导入失败',
                    text: '配置文件格式错误',
                    icon: 'error'
                });
            }
        };
        reader.readAsText(file);
    };
    input.click();
}

// 保存参数
function saveParameter(btn) {
    const row = btn.closest('tr');
    const input = row.querySelector('input');
    const value = parseFloat(input.value);
    const name = row.querySelector('td:first-child span').textContent.trim();
    const originalValue = input.dataset.original;
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    const step = parseFloat(input.step);

    // 验证输入是否为有效数字
    if (isNaN(value)) {
        Swal.fire({
            title: '输入错误',
            text: `请输入有效的数字`,
            icon: 'error'
        });
        input.value = originalValue;
        return;
    }

    // 验证范围
    if (value < min || value > max) {
        Swal.fire({
            title: '参数错误',
            html: `
                <div class="text-start">
                    <p class="mb-2">${name}的值超出有效范围：</p>
                    <ul>
                        <li>当前输入值：${value}</li>
                        <li>有效范围：${min} 到 ${max}</li>
                    </ul>
                    <p class="mb-0">已自动恢复为原始值</p>
                </div>
            `,
            icon: 'error'
        });
        input.value = originalValue;
        return;
    }

    // 验证步长
    const stepError = Math.abs((value - min) % step);
    if (stepError > Number.EPSILON) {
        const correctedValue = Math.round((value - min) / step) * step + min;
        Swal.fire({
            title: '参数错误',
            html: `
                <div class="text-start">
                    <p class="mb-2">${name}的值必须是 ${step} 的整数倍：</p>
                    <ul>
                        <li>当前输入值：${value}</li>
                        <li>建议修正为：${correctedValue}</li>
                    </ul>
                    <p class="mb-0">是否使用修正后的值？</p>
                </div>
            `,
            icon: 'warning',
            showCancelButton: true,
            confirmButtonText: '使用修正值',
            cancelButtonText: '恢复原值'
        }).then((result) => {
            if (result.isConfirmed) {
                input.value = correctedValue;
                saveParameterValue(input, correctedValue, name, originalValue);
            } else {
                input.value = originalValue;
            }
        });
        return;
    }

    // 如果验证通过，继续保存操作
    saveParameterValue(input, value, name, originalValue);
}

// 辅助函数：保存参数值
function saveParameterValue(input, value, name, originalValue) {
    Swal.fire({
        title: '确认保存',
        html: `
            <div class="text-start">
                <p class="mb-2">是否保存以下修改：</p>
                <div class="d-flex justify-content-between border-bottom py-2">
                    <span class="text-muted">参数名称：</span>
                    <span class="fw-bold">${name}</span>
                </div>
                <div class="d-flex justify-content-between border-bottom py-2">
                    <span class="text-muted">原始值：</span>
                    <span>${originalValue}</span>
                </div>
                <div class="d-flex justify-content-between py-2">
                    <span class="text-muted">新值：</span>
                    <span class="text-primary">${value}</span>
                </div>
            </div>
        `,
        icon: 'question',
        showCancelButton: true,
        confirmButtonText: '保存',
        cancelButtonText: '取消'
    }).then((result) => {
        if (result.isConfirmed) {
            input.dataset.original = value;
            Swal.fire({
                title: '保存成功',
                text: `${name}已更新为${value}`,
                icon: 'success',
                toast: true,
                position: 'top-end',
                showConfirmButton: false,
                timer: 3000
            });
        } else {
            input.value = originalValue;
        }
    });
}

// 重置参数
function resetParameter(btn) {
    const row = btn.closest('tr');
    const input = row.querySelector('input');
    const name = row.cells[0].textContent;

    Swal.fire({
        title: '确认重置',
        text: `是否将${name}重置为默认值？`,
        icon: 'warning',
        showCancelButton: true,
        confirmButtonText: '重置',
        cancelButtonText: '取消'
    }).then((result) => {
        if (result.isConfirmed) {
            // 重置为默认值
            const defaultValues = {
                '时序窗口大小': '168',
                '注意力头数': '8',
                '学习率': '0.001'
            };
            input.value = defaultValues[name];
            Swal.fire({
                title: '重置成功',
                text: '参数已恢复默认值',
                icon: 'success'
            });
        }
    });
}

// 删除策略
function deleteStrategy(id) {
    Swal.fire({
        title: '确认删除',
        text: '删除后无法恢复，是否继续？',
        icon: 'warning',
        showCancelButton: true,
        confirmButtonText: '删除',
        cancelButtonText: '取消'
    }).then((result) => {
        if (result.isConfirmed) {
            // TODO: 调用API删除策略
            Swal.fire({
                title: '删除成功',
                text: '预测策略已删除',
                icon: 'success'
            });
        }
    });
}

// 删除角色
function deleteRole(id) {
    Swal.fire({
        title: '确认删除',
        text: '删除角色可能影响现有用户权限，是否继续？',
        icon: 'warning',
        showCancelButton: true,
        confirmButtonText: '删除',
        cancelButtonText: '取消'
    }).then((result) => {
        if (result.isConfirmed) {
            // TODO: 调用API删除角色
            Swal.fire({
                title: '删除成功',
                text: '角色已删除',
                icon: 'success'
            });
        }
    });
}

// 初始化图表
function initializeCharts() {
    const chartOptions = {
        series: [{
            name: '告警次数',
            data: [3, 7, 4, 9, 5, 2, 3]
        }],
        chart: {
            type: 'area',
            height: 120,
            sparkline: {
                enabled: true
            },
            toolbar: {
                show: false
            }
        },
        stroke: {
            curve: 'smooth',
            width: 2
        },
        fill: {
            type: 'gradient',
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.5,
                opacityTo: 0.1
            }
        },
        colors: ['#4F8CBE'],
        tooltip: {
            fixed: {
                enabled: false
            },
            x: {
                show: false
            },
            y: {
                title: {
                    formatter: function (seriesName) {
                        return '告警次数：'
                    }
                }
            },
            marker: {
                show: false
            }
        }
    };

    // 初始化各个图表
    new ApexCharts(document.querySelector("#windChart"), {
        ...chartOptions,
        series: [{
            name: '告警次数',
            data: [3, 7, 4, 9, 5, 2, 3]
        }]
    }).render();

    new ApexCharts(document.querySelector("#loadChart"), {
        ...chartOptions,
        series: [{
            name: '告警次数',
            data: [2, 4, 3, 1, 5, 3, 2]
        }]
    }).render();

    new ApexCharts(document.querySelector("#confidenceChart"), {
        ...chartOptions,
        series: [{
            name: '告警次数',
            data: [5, 3, 6, 4, 2, 3, 4]
        }]
    }).render();
}

// 批量更新阈值
function batchUpdate() {
    Swal.fire({
        title: '批量更新阈值',
        html: `
            <form id="batchUpdateForm">
                <div class="mb-3">
                    <label class="form-label">选择更新项</label>
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" id="updateWind" checked>
                        <label class="form-check-label" for="updateWind">
                            风电产能波动阈值
                        </label>
                    </div>
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" id="updateLoad" checked>
                        <label class="form-check-label" for="updateLoad">
                            负荷预测偏差阈值
                        </label>
                    </div>
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" id="updateConfidence" checked>
                        <label class="form-check-label" for="updateConfidence">
                            时序预测置信度阈值
                        </label>
                    </div>
                </div>
                <div class="mb-3">
                    <label class="form-label">调整方式</label>
                    <select class="form-select" id="adjustmentType">
                        <option value="percent">百分比调整</option>
                        <option value="absolute">固定值调整</option>
                        <option value="optimal">优化建议值</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label class="form-label">调整值</label>
                    <div class="input-group">
                        <input type="number" class="form-control" id="adjustmentValue" value="10">
                        <span class="input-group-text">%</span>
                    </div>
                </div>
            </form>
        `,
        showCancelButton: true,
        confirmButtonText: '更新',
        cancelButtonText: '取消',
        preConfirm: () => {
            return {
                wind: document.getElementById('updateWind').checked,
                load: document.getElementById('updateLoad').checked,
                confidence: document.getElementById('updateConfidence').checked,
                type: document.getElementById('adjustmentType').value,
                value: document.getElementById('adjustmentValue').value
            }
        }
    }).then((result) => {
        if (result.isConfirmed) {
            // 执行批量更新
            updateThresholds(result.value);
        }
    });
}

// 重置所有阈值
function resetAllThresholds() {
    Swal.fire({
        title: '确认重置',
        text: '是否将所有阈值恢复为默认值？',
        icon: 'warning',
        showCancelButton: true,
        confirmButtonText: '重置',
        cancelButtonText: '取消'
    }).then((result) => {
        if (result.isConfirmed) {
            // 重置所有滑块到默认值
            document.getElementById('windSlider').noUiSlider.set(15);
            document.getElementById('loadSlider').noUiSlider.set(10);
            document.getElementById('confidenceSlider').noUiSlider.set(85);

            Swal.fire({
                title: '重置成功',
                text: '所有阈值已恢复默认值',
                icon: 'success'
            });
        }
    });
}

// 查看告警历史
function viewAlertHistory(type) {
    const typeNames = {
        wind: '风电产能波动',
        load: '负荷预测偏差',
        confidence: '时序预测置信度'
    };

    Swal.fire({
        title: `${typeNames[type]}告警历史`,
        html: `
            <div class="table-responsive">
                <table class="table table-hover">
                    <thead>
                        <tr>
                            <th>时间</th>
                            <th>告警值</th>
                            <th>阈值</th>
                            <th>状态</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>2024-03-21 15:30</td>
                            <td>25.6%</td>
                            <td>20%</td>
                            <td><span class="badge bg-danger">已触发</span></td>
                        </tr>
                        <tr>
                            <td>2024-03-21 14:20</td>
                            <td>18.3%</td>
                            <td>20%</td>
                            <td><span class="badge bg-success">已恢复</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        `,
        width: '600px'
    });
}

// 更新阈值
function updateThresholds(config) {
    const updates = [];
    const sliders = {
        wind: { min: 0, max: 30, current: 15 },
        load: { min: 0, max: 20, current: 10 },
        confidence: { min: 0, max: 100, current: 85 }
    };

    // 计算新值的函数
    const calculateNewValue = (type, value, adjustmentType, adjustmentValue) => {
        const slider = sliders[type];
        let newValue;
        
        if (adjustmentType === 'percent') {
            // 百分比调整
            newValue = slider.current * (1 + adjustmentValue / 100);
        } else if (adjustmentType === 'absolute') {
            // 固定值调整
            newValue = slider.current + parseFloat(adjustmentValue);
        } else if (adjustmentType === 'optimal') {
            // 优化建议值
            const optimalValues = {
                wind: 18,      // 建议的风电阈值
                load: 12,      // 建议的负荷阈值
                confidence: 90  // 建议的置信度阈值
            };
            newValue = optimalValues[type];
        }

        // 确保值在有效范围内
        return Math.min(Math.max(newValue, slider.min), slider.max);
    };

    // 更新每个选中的滑块
    Object.entries({
        wind: config.wind,
        load: config.load,
        confidence: config.confidence
    }).forEach(([type, isSelected]) => {
        if (!isSelected) return;

        const newValue = calculateNewValue(
            type,
            document.getElementById(`${type}Input`).value,
            config.type,
            config.value
        );

        // 更新滑块
        const slider = document.getElementById(`${type}Slider`);
        if (slider && slider.noUiSlider) {
            slider.noUiSlider.set(newValue);
        }

        // 更新输入框
        const input = document.getElementById(`${type}Input`);
        if (input) {
            input.value = type === 'confidence' ? 
                Math.round(newValue) : 
                newValue.toFixed(1);
        }

        updates.push(`${
            type === 'wind' ? '风电产能波动阈值' :
            type === 'load' ? '负荷预测偏差阈值' :
            '时序预测置信度阈值'
        }`);
    });

    if (updates.length > 0) {
        Swal.fire({
            title: '更新成功',
            html: `
                <div class="text-start">
                    <p class="mb-2">已更新以下阈值：</p>
                    <ul class="list-unstyled">
                        ${updates.map(name => `
                            <li><i class="fas fa-check text-success me-2"></i>${name}</li>
                        `).join('')}
                    </ul>
                    <p class="mt-2 mb-0 text-muted">
                        <i class="fas fa-info-circle me-1"></i>
                        所有更新已应用并保存
                    </p>
                </div>
            `,
            icon: 'success'
        });
    }
}

// 同步配置
function syncConfig() {
    Swal.fire({
        title: '同步配置',
        html: `
            <div class="text-start">
                <p class="mb-2">选择同步目标：</p>
                <div class="form-check mb-2">
                    <input class="form-check-input" type="checkbox" id="syncProd" checked>
                    <label class="form-check-label" for="syncProd">
                        生产环境
                    </label>
                </div>
                <div class="form-check mb-2">
                    <input class="form-check-input" type="checkbox" id="syncTest">
                    <label class="form-check-label" for="syncTest">
                        测试环境
                    </label>
                </div>
                <div class="form-check">
                    <input class="form-check-input" type="checkbox" id="syncDev">
                    <label class="form-check-label" for="syncDev">
                        开发环境
                    </label>
                </div>
            </div>
        `,
        showCancelButton: true,
        confirmButtonText: '同步',
        cancelButtonText: '取消'
    }).then((result) => {
        if (result.isConfirmed) {
            Swal.fire({
                title: '同步成功',
                text: '配置已同步到选定环境',
                icon: 'success'
            });
        }
    });
}

// 比较版本
function compareVersions() {
    Swal.fire({
        title: '版本对比',
        html: `
            <div class="row mb-3">
                <div class="col-6">
                    <label class="form-label">基准版本</label>
                    <select class="form-select">
                        <option value="v2.1.3" selected>v2.1.3 (当前)</option>
                        <option value="v2.1.2">v2.1.2</option>
                        <option value="v2.1.1">v2.1.1</option>
                    </select>
                </div>
                <div class="col-6">
                    <label class="form-label">对比版本</label>
                    <select class="form-select">
                        <option value="v2.1.2" selected>v2.1.2</option>
                        <option value="v2.1.1">v2.1.1</option>
                        <option value="v2.1.0">v2.1.0</option>
                    </select>
                </div>
            </div>
            <div class="table-responsive">
                <table class="table table-hover">
                    <thead>
                        <tr>
                            <th>参数</th>
                            <th>v2.1.3</th>
                            <th>v2.1.2</th>
                            <th>差异</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>风电产能波动阈值</td>
                            <td>15%</td>
                            <td>20%</td>
                            <td><span class="text-danger">-5%</span></td>
                        </tr>
                        <tr>
                            <td>负荷预测偏差阈值</td>
                            <td>10%</td>
                            <td>10%</td>
                            <td><span class="text-muted">无变化</span></td>
                        </tr>
                        <tr>
                            <td>时序预测置信度阈值</td>
                            <td>85%</td>
                            <td>80%</td>
                            <td><span class="text-success">+5%</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        `,
        width: '800px'
    });
}

// 阈值管理对象
const thresholdManager = {
    charts: {}, // 存储图表实例
    
    config: {
        wind: {
            sliderId: 'windSlider',
            inputId: 'windInput',
            chartId: 'windChart',
            min: 0,
            max: 30,
            current: 15,
            step: 0.1,
            title: '风电产能波动阈值',
            color: '#4F8CBE'
        },
        load: {
            sliderId: 'loadSlider',
            inputId: 'loadInput',
            chartId: 'loadChart',
            min: 0,
            max: 20,
            current: 10,
            step: 0.1,
            title: '负荷预测偏差阈值',
            color: '#2ab57d'
        },
        confidence: {
            sliderId: 'confidenceSlider',
            inputId: 'confidenceInput',
            chartId: 'confidenceChart',
            min: 0,
            max: 100,
            current: 85,
            step: 1,
            title: '时序预测置信度阈值',
            color: '#fd625e'
        }
    },

    // 初始化所有阈值设置
    init() {
        console.log('Initializing threshold manager...');
        Object.keys(this.config).forEach(key => {
            this.initThreshold(key);
        });
    },

    // 初始化单个阈值设置
    initThreshold(type) {
        console.log(`Initializing threshold for ${type}...`);
        const config = this.config[type];
        
        // 初始化图表
        try {
            const chartDom = document.getElementById(config.chartId);
            if (!chartDom) {
                console.error(`Chart element ${config.chartId} not found`);
                return;
            }

            // 销毁已存在的图表实例
            if (this.charts[type]) {
                this.charts[type].dispose();
            }

            // 创建新的图表实例
            const chart = echarts.init(chartDom, 'dark');
            this.charts[type] = chart;

            const option = {
                backgroundColor: 'transparent',
                series: [{
                    type: 'gauge',
                    startAngle: 180,
                    endAngle: 0,
                    min: config.min,
                    max: config.max,
                    splitNumber: 2,
                    radius: '100%',
                    itemStyle: {
                        color: config.color
                    },
                    progress: {
                        show: true,
                        roundCap: true,
                        width: 12,
                        itemStyle: {
                            color: config.color
                        }
                    },
                    pointer: {
                        show: false
                    },
                    axisLine: {
                        roundCap: true,
                        lineStyle: {
                            width: 12,
                            color: [[1, 'rgba(255, 255, 255, 0.1)']]
                        }
                    },
                    axisTick: {
                        show: false
                    },
                    splitLine: {
                        show: false
                    },
                    axisLabel: {
                        show: true,
                        distance: -20,
                        color: '#999',
                        fontSize: 14,
                        formatter: function(value) {
                            if (value === config.min) {
                                return `{minValue|${value}%}`;
                            }
                            if (value === config.max) {
                                return `{maxValue|${value}%}`;
                            }
                            return '';
                        },
                        rich: {
                            minValue: {
                                align: 'left',
                                padding: [0, 0, -5, -25]
                            },
                            maxValue: {
                                align: 'right',
                                padding: [0, -25, -5, 0]
                            }
                        }
                    },
                    detail: {
                        show: true,
                        valueAnimation: true,
                        fontSize: 24,
                        offsetCenter: [0, '20%'],
                        formatter: '{value}%',
                        color: '#fff'
                    },
                    data: [{
                        value: config.current,
                        name: config.title
                    }]
                }]
            };
            
            chart.setOption(option);
            console.log(`Chart ${config.chartId} initialized`);

            // 监听窗口大小变化
            window.addEventListener('resize', () => {
                chart.resize();
            });

        } catch (error) {
            console.error(`Error initializing chart for ${type}:`, error);
        }

        // 初始化滑块
        try {
            const slider = document.getElementById(config.sliderId);
            if (!slider) {
                console.error(`Slider element ${config.sliderId} not found`);
                return;
            }

            if (slider.noUiSlider) {
                slider.noUiSlider.destroy();
            }

            noUiSlider.create(slider, {
                start: [config.current],
                connect: [true, false],
                step: config.step,
                range: {
                    'min': config.min,
                    'max': config.max
                }
            });

            // 绑定滑块事件
            slider.noUiSlider.on('update', (values) => {
                const value = parseFloat(values[0]);
                const input = document.getElementById(config.inputId);
                if (input) {
                    input.value = value.toFixed(config.step < 1 ? 1 : 0);
                }
                this.updateChart(type, value);
            });

            console.log(`Slider ${config.sliderId} initialized`);

        } catch (error) {
            console.error(`Error initializing slider for ${type}:`, error);
        }

        // 绑定输入框事件
        try {
            const input = document.getElementById(config.inputId);
            if (!input) {
                console.error(`Input element ${config.inputId} not found`);
                return;
            }

            input.value = config.current;
            
            // 添加输入事件监听
            input.addEventListener('input', (e) => {
                let value = parseFloat(e.target.value);
                
                // 检查是否为有效数字
                if (isNaN(value)) {
                    e.target.value = config.current;
                    showToast('输入错误', '请输入有效的数字', 'error');
                    return;
                }
                
                // 检查是否超出范围
                if (value < config.min || value > config.max) {
                    value = Math.min(Math.max(value, config.min), config.max);
                    e.target.value = value;
                    showToast('超出范围', `有效范围为 ${config.min} - ${config.max}，已自动调整`, 'warning');
                }
                
                // 更新滑块和图表
                const slider = document.getElementById(config.sliderId);
                if (slider && slider.noUiSlider) {
                    slider.noUiSlider.set(value);
                }
                this.updateChart(type, value);
            });

            // 添加失焦事件监听
            input.addEventListener('blur', (e) => {
                let value = parseFloat(e.target.value);
                if (!isNaN(value)) {
                    const step = config.step;
                    const min = config.min;
                    const stepsFromMin = Math.round((value - min) / step);
                    value = min + (stepsFromMin * step);
                    value = parseFloat(value.toFixed(step < 1 ? 1 : 0));
                    
                    e.target.value = value;
                    const slider = document.getElementById(config.sliderId);
                    if (slider && slider.noUiSlider) {
                        slider.noUiSlider.set(value);
                    }
                    this.updateChart(type, value);
                }
            });

            console.log(`Input ${config.inputId} initialized`);

        } catch (error) {
            console.error(`Error initializing input for ${type}:`, error);
        }
    },

    // 更新图表显示
    updateChart(type, value) {
        try {
            const chart = this.charts[type];
            if (!chart) {
                console.error(`Chart instance for ${type} not found`);
                return;
            }

            chart.setOption({
                series: [{
                    data: [{
                        value: value
                    }]
                }]
            });

            console.log(`Chart ${type} updated with value: ${value}`);

        } catch (error) {
            console.error(`Error updating chart for ${type}:`, error);
        }
    }
};

// 确保在 DOM 和所有资源加载完成后初始化
window.addEventListener('load', () => {
    console.log('Window loaded, checking dependencies...');
    
    // 检查依赖是否加载
    if (typeof echarts === 'undefined') {
        console.error('ECharts is not loaded!');
        return;
    }
    if (typeof noUiSlider === 'undefined') {
        console.error('noUiSlider is not loaded!');
        return;
    }

    // 初始化阈值管理器
    console.log('Dependencies loaded, initializing threshold manager...');
    thresholdManager.init();

    // 添加调试信息
    console.log('Threshold manager initialization completed');
});

// 全局配置对象
const CONFIG = {
    version: 'v2.1.3',
    configHistory: [
        {
            version: 'v2.1.3',
            date: '2024-03-21 15:30',
            operator: '管理员',
            description: '更新预警阈值配置',
            config: {
                thresholds: {
                    windThreshold: 15,
                    loadThreshold: 10,
                    confidenceThreshold: 85
                },
                modelParams: {
                    // Basic
                    windowSize: 168,
                    attentionHeads: 8,
                    learningRate: 0.001,
                    // Advanced
                    hiddenDim: 256,
                    dropoutRate: 0.1,
                    // Optimization
                    batchSize: 32,
                    warmupSteps: 1000
                },
                alertSettings: {
                    level: 'warning',
                    methods: {
                        email: true,
                        sms: true,
                        system: true
                    },
                    time: 'always'
                }
            }
        },
        {
            version: 'v2.1.2',
            date: '2024-03-21 14:20',
            operator: '管理员',
            description: '导入生产环境配置',
            config: {
                thresholds: {
                    windThreshold: 20,
                    loadThreshold: 12,
                    confidenceThreshold: 80
                },
                modelParams: {
                     // Basic
                    windowSize: 144,
                    attentionHeads: 6,
                    learningRate: 0.002,
                    // Advanced
                    hiddenDim: 128,
                    dropoutRate: 0.2,
                    // Optimization
                    batchSize: 64,
                    warmupSteps: 500
                },
                alertSettings: {
                    level: 'info',
                    methods: {
                        email: true,
                        sms: false,
                        system: true
                    },
                    time: 'workday'
                }
            }
        }
    ]
};

// 全局状态对象
const STATE = {
    isDirty: false,
    pendingChanges: [],
    lastSyncTime: new Date().toISOString()
};

// 初始化函数
function initializeSystem() {
    initializeSliders();
    initializeModelParams(); // New function to load params
    initializeFormValidation();
    initializeCharts();
    initializeTooltips();
    headerInit.init(); // Initialize header elements like dropdowns
}

// 初始化模型参数输入框
function initializeModelParams() {
    const currentParams = CONFIG.configHistory[0].config.modelParams;
    const paramInputs = document.querySelectorAll('#modelParamContent input[data-param-key]');

    paramInputs.forEach(input => {
        const paramKey = input.getAttribute('data-param-key');
        if (currentParams.hasOwnProperty(paramKey)) {
            input.value = currentParams[paramKey];
            input.setAttribute('data-original', currentParams[paramKey]); // Set original value for reset
        }

        // Remove previous listener if any (safer if this code runs multiple times)
        input.removeEventListener('blur', validateParamInput);
        input.removeEventListener('change', validateParamInput);
        
        // Add change event listener for validation
        input.addEventListener('change', validateParamInput);
        // *** DEBUG ALERT 1 ***
        // alert(`Listener attached for: ${paramKey}`); 
        // Temporarily disabled alert as it can be annoying for many inputs
        // console.log(`Listener attached for: ${paramKey}`); // Use console log instead

    });
    console.log("Model parameters initialized and change listeners attached.");
}

// Validate parameter input on change (previously on blur)
function validateParamInput(event) {
    // *** DEBUG ALERT 2 ***
    alert(`validateParamInput called for: ${event.target.getAttribute('data-param-key')}`); 
    
    const input = event.target;
    const paramKey = input.getAttribute('data-param-key');
    const valueStr = input.value;
    const minStr = input.min;
    const maxStr = input.max;

    // Debugging logs (keep them for now)
    console.log(`validateParamInput triggered for ${paramKey} via ${event.type}`);
    console.log(`  Input value (string): '${valueStr}'`);
    console.log(`  Min attribute: '${minStr}', Max attribute: '${maxStr}'`);

    // Try parsing
    const value = parseFloat(valueStr);
    const min = parseFloat(minStr);
    const max = parseFloat(maxStr);

    console.log(`  Parsed values: value=${value}, min=${min}, max=${max}`);

    if (isNaN(value) || isNaN(min) || isNaN(max)) {
        console.log('  Validation skipped: Failed to parse numeric values.');
        // Optionally reset to original value if parsing fails
        // input.value = input.getAttribute('data-original');
        // Show an error toast?
        // Swal.fire({ toast: true, position: 'top-end', icon: 'error', title: '无效输入', text: `参数 ${paramKey} 必须是数字。`, showConfirmButton: false, timer: 3000 });
        return; 
    }

    let correctedValue = value;
    let message = '';

    if (value < min) {
        console.log(`  Condition met: value (${value}) < min (${min})`);
        correctedValue = min;
        message = `参数 ${paramKey} 的值低于最小值 (${min})，已自动修正。`;
    } else if (value > max) {
        console.log(`  Condition met: value (${value}) > max (${max})`);
        correctedValue = max;
        message = `参数 ${paramKey} 的值超过最大值 (${max})，已自动修正。`;
    } else {
        console.log(`  Value ${value} is within range [${min}, ${max}]. No correction needed.`);
    }

    console.log(`  Original value from input: ${value}, Corrected value: ${correctedValue}`);

    // Use correctedValue for comparison and assignment
    if (correctedValue !== value) {
        console.log(`  Applying correction: Setting input value to ${correctedValue}`);
        input.value = correctedValue; 
        // Show a toast notification
        Swal.fire({
            toast: true,
            position: 'top-end',
            icon: 'warning',
            title: '输入值已调整',
            text: message,
            showConfirmButton: false,
            timer: 3500,
            timerProgressBar: true
        });
    } else {
        console.log('  No update applied as correctedValue is the same as the parsed input value.');
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', initializeSystem);

// 保存配置 (下载为 JSON)
function saveConfig() {
    console.log("Attempting to save comprehensive configuration...");
    try {
        // 1. Gather Thresholds (Value and Enabled Status)
        const thresholds = {
            wind: {
                value: parseFloat(document.getElementById('windSlider').noUiSlider.get()),
                enabled: document.getElementById('windAlertSwitch')?.checked || false
            },
            load: {
                value: parseFloat(document.getElementById('loadSlider').noUiSlider.get()),
                enabled: document.getElementById('loadAlertSwitch')?.checked || false
            },
            confidence: {
                value: parseFloat(document.getElementById('confidenceSlider').noUiSlider.get()),
                enabled: document.getElementById('confidenceAlertSwitch')?.checked || false
            }
        };
        console.log("Thresholds gathered:", thresholds);

        // 2. Gather Model Parameters
        const modelParams = {};
        const paramInputs = document.querySelectorAll('#modelParamContent input[data-param-key]');
        paramInputs.forEach(input => {
            const key = input.getAttribute('data-param-key');
            // Parse numbers, keep others as strings (adjust if other types needed)
            const value = input.value;
            const parsedValue = parseFloat(value);
            modelParams[key] = isNaN(parsedValue) ? value : parsedValue; 
        });
        console.log("Model Params gathered:", modelParams);

        // 3. Gather Alert Settings (Level, Methods, Time)
        const alertSettings = {
            level: document.getElementById('alertLevelSelect')?.value || 'warning',
            methods: {
                email: document.getElementById('alertEmail')?.checked || false,
                sms: document.getElementById('alertSMS')?.checked || false,
                system: document.getElementById('alertSystem')?.checked || false
            },
            time: document.getElementById('alertTimeSelect')?.value || 'always'
        };
         console.log("Alert Settings gathered:", alertSettings);

        // 4. Combine into final configuration object
        const comprehensiveConfig = {
            version: CONFIG.version, // Still useful to know the base version
            savedAt: new Date().toISOString(),
            thresholdSettings: thresholds, // Renamed for clarity
            modelParameters: modelParams,    // Renamed for clarity
            alertConfiguration: alertSettings // Renamed for clarity
            // Add other sections here if needed in the future
        };
        console.log("Final comprehensive config object:", comprehensiveConfig);

        // 5. Generate JSON and trigger download
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(comprehensiveConfig, null, 2));
        const downloadAnchorNode = document.createElement('a');
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", `system_config_${timestamp}.json`);
        document.body.appendChild(downloadAnchorNode); // required for firefox
        downloadAnchorNode.click();
        downloadAnchorNode.remove();

        Swal.fire({
            title: '配置已保存',
            text: '当前系统配置已导出为 JSON 文件。',
            icon: 'success',
            timer: 2500,
            showConfirmButton: false
        });

    } catch (error) {
        console.error("Error saving configuration:", error);
        Swal.fire(
            '保存失败',
            '导出配置文件时出错，请检查控制台获取更多信息。',
            'error'
        );
    }
}

// Header 初始化
const headerInit = {
    init() {
        this.initFeatherIcons();
        this.initLayoutMode();
        this.initDropdowns();
        this.initSearch();
        this.initNotifications();
    },

    // 初始化 Feather 图标
    initFeatherIcons() {
        if (typeof feather !== 'undefined') {
            feather.replace();
        }
    },

    // 初始化布局模式
    initLayoutMode() {
        const modeBtn = document.getElementById('mode-setting-btn');
        if (modeBtn) {
            modeBtn.addEventListener('click', function() {
                const body = document.body;
                if (body.getAttribute('data-layout-mode') === 'dark') {
                    body.setAttribute('data-layout-mode', 'light');
                } else {
                    body.setAttribute('data-layout-mode', 'dark');
                }
            });
        }
    },

    // 初始化下拉菜单
    initDropdowns() {
        // 语言切换
        const langItems = document.querySelectorAll('.language');
        langItems.forEach(item => {
            item.addEventListener('click', function(e) {
                e.preventDefault();
                const lang = this.getAttribute('data-lang');
                const img = document.getElementById('header-lang-img');
                if (img) {
                    img.src = `static/picture/${lang}.jpg`;
                }
            });
        });
    },

    // 初始化搜索功能
    initSearch() {
        const searchDropdown = document.getElementById('page-header-search-dropdown');
        if (searchDropdown) {
            searchDropdown.addEventListener('click', function(e) {
                e.preventDefault();
                document.querySelector('.app-search').classList.toggle('d-none');
            });
        }
    },

    // 初始化通知功能
    initNotifications() {
        const notificationDropdown = document.getElementById('page-header-notifications-dropdown');
        if (notificationDropdown) {
            notificationDropdown.addEventListener('click', function(e) {
                e.preventDefault();
                // 这里可以添加获取最新通知的逻辑
            });
        }
    }
};

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    // 初始化配置管理
    configManagement.init();
    
    // 初始化头部功能
    headerInit.init();
});

// 查看历史版本
function viewConfigHistory() {
    Swal.fire({
        title: '配置历史',
        html: `
            <div class="table-responsive">
                <table class="table table-hover">
                    <thead>
                        <tr>
                            <th>版本</th>
                            <th>时间</th>
                            <th>操作人</th>
                            <th>说明</th>
                            <th>操作</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${CONFIG.configHistory.map(history => `
                            <tr>
                                <td>${history.version}</td>
                                <td>${history.date}</td>
                                <td>${history.operator}</td>
                                <td>${history.description}</td>
                                <td>
                                    <div class="btn-group">
                                        <button class="btn btn-sm btn-outline-primary" onclick="window.viewHistoryDetail('${history.version}')">
                                            <i class="fas fa-eye"></i>
                                        </button>
                                        <button class="btn btn-sm btn-outline-warning" onclick="window.rollbackConfig('${history.version}')">
                                            <i class="fas fa-history"></i> 回溯
                                        </button>
                                    </div>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `,
        width: '800px',
        showCloseButton: true,
        showConfirmButton: false
    });
}

// 查看历史版本详情
function viewHistoryDetail(version) {
    const history = CONFIG.configHistory.find(h => h.version === version);
    if (!history) return;

    Swal.fire({
        title: `配置详情 - ${version}`,
        html: `
            <div class="accordion" id="historyDetailAccordion">
                <!-- 基本信息 -->
                <div class="accordion-item">
                    <h2 class="accordion-header">
                        <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#basicInfo">
                            基本信息
                        </button>
                    </h2>
                    <div id="basicInfo" class="accordion-collapse collapse show" data-bs-parent="#historyDetailAccordion">
                        <div class="accordion-body">
                            <table class="table table-sm">
                                <tr>
                                    <th width="30%">版本号</th>
                                    <td>${history.version}</td>
                                </tr>
                                <tr>
                                    <th>更新时间</th>
                                    <td>${history.date}</td>
                                </tr>
                                <tr>
                                    <th>操作人</th>
                                    <td>${history.operator}</td>
                                </tr>
                                <tr>
                                    <th>变更说明</th>
                                    <td>${history.description}</td>
                                </tr>
                            </table>
                        </div>
                    </div>
                </div>

                <!-- 阈值配置 -->
                <div class="accordion-item">
                    <h2 class="accordion-header">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#thresholdInfo">
                            阈值配置
                        </button>
                    </h2>
                    <div id="thresholdInfo" class="accordion-collapse collapse" data-bs-parent="#historyDetailAccordion">
                        <div class="accordion-body">
                            <table class="table table-sm">
                                <tr>
                                    <th width="30%">风电产能波动阈值</th>
                                    <td>${history.config.thresholds.windThreshold}%</td>
                                </tr>
                                <tr>
                                    <th>负荷预测偏差阈值</th>
                                    <td>${history.config.thresholds.loadThreshold}%</td>
                                </tr>
                                <tr>
                                    <th>时序预测置信度阈值</th>
                                    <td>${history.config.thresholds.confidenceThreshold}%</td>
                                </tr>
                            </table>
                        </div>
                    </div>
                </div>

                <!-- 模型参数 -->
                <div class="accordion-item">
                    <h2 class="accordion-header">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#modelInfo">
                            模型参数
                        </button>
                    </h2>
                    <div id="modelInfo" class="accordion-collapse collapse" data-bs-parent="#historyDetailAccordion">
                        <div class="accordion-body">
                            <table class="table table-sm">
                                <tr>
                                    <th width="30%">时序窗口大小</th>
                                    <td>${history.config.modelParams.windowSize}</td>
                                </tr>
                                <tr>
                                    <th>注意力头数</th>
                                    <td>${history.config.modelParams.attentionHeads}</td>
                                </tr>
                                <tr>
                                    <th>学习率</th>
                                    <td>${history.config.modelParams.learningRate}</td>
                                </tr>
                            </table>
                        </div>
                    </div>
                </div>

                <!-- 告警设置 -->
                <div class="accordion-item">
                    <h2 class="accordion-header">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#alertInfo">
                            告警设置
                        </button>
                    </h2>
                    <div id="alertInfo" class="accordion-collapse collapse" data-bs-parent="#historyDetailAccordion">
                        <div class="accordion-body">
                            <table class="table table-sm">
                                <tr>
                                    <th width="30%">告警级别</th>
                                    <td>${history.config.alertSettings.level}</td>
                                </tr>
                                <tr>
                                    <th>告警方式</th>
                                    <td>
                                        ${history.config.alertSettings.methods.email ? '邮件通知<br>' : ''}
                                        ${history.config.alertSettings.methods.sms ? '短信通知<br>' : ''}
                                        ${history.config.alertSettings.methods.system ? '系统通知' : ''}
                                    </td>
                                </tr>
                                <tr>
                                    <th>告警时间</th>
                                    <td>${history.config.alertSettings.time}</td>
                                </tr>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        `,
        width: '800px',
        showCloseButton: true,
        showConfirmButton: false
    });
}

// 回溯配置
function rollbackConfig(version) {
    const history = CONFIG.configHistory.find(h => h.version === version);
    if (!history) return;

    Swal.fire({
        title: '确认回溯',
        html: `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                您确定要将配置回溯到版本 ${version} 吗？
            </div>
            <div class="text-start">
                <p class="mb-2">此操作将：</p>
                <ul>
                    <li>恢复所有阈值设置</li>
                    <li>恢复所有模型参数</li>
                    <li>恢复告警配置</li>
                </ul>
                <p class="mb-0 text-danger">
                    <i class="fas fa-info-circle me-1"></i>
                    回溯后将创建新的版本记录，您可以随时切换回当前版本
                </p>
            </div>
        `,
        icon: 'warning',
        showCancelButton: true,
        confirmButtonText: '确认回溯',
        cancelButtonText: '取消',
        confirmButtonColor: '#d33'
    }).then((result) => {
        if (result.isConfirmed) {
            // 执行回溯操作
            const oldConfig = CONFIG.configHistory[0].config; // 保存当前配置
            
            // 更新阈值
            document.getElementById('windSlider').noUiSlider.set(history.config.thresholds.windThreshold);
            document.getElementById('loadSlider').noUiSlider.set(history.config.thresholds.loadThreshold);
            document.getElementById('confidenceSlider').noUiSlider.set(history.config.thresholds.confidenceThreshold);
            
            // 更新模型参数
            document.getElementById('windowSizeInput').value = history.config.modelParams.windowSize;
            document.getElementById('attentionHeadsInput').value = history.config.modelParams.attentionHeads;
            document.getElementById('learningRateInput').value = history.config.modelParams.learningRate;
            
            // 更新告警设置
            document.getElementById('alertLevel').value = history.config.alertSettings.level;
            document.getElementById('alertEmail').checked = history.config.alertSettings.methods.email;
            document.getElementById('alertSMS').checked = history.config.alertSettings.methods.sms;
            document.getElementById('alertSystem').checked = history.config.alertSettings.methods.system;
            document.getElementById('alertTime').value = history.config.alertSettings.time;

            // 创建新的版本记录
            const newVersion = {
                version: `v${parseFloat(version.substring(1)) + 0.0001}`,
                date: new Date().toLocaleString('sv-SE'),
                operator: '当前用户',
                description: `回溯自 ${version}`,
                config: history.config
            };

            // 在历史记录开头插入新版本
            CONFIG.configHistory.unshift(newVersion);
            
            // 保存旧配置到历史记录
            CONFIG.configHistory.splice(1, 0, {
                version: CONFIG.version,
                date: new Date().toLocaleString('sv-SE'),
                operator: '当前用户',
                description: '回溯前的配置',
                config: oldConfig
            });

            // 更新当前版本
            CONFIG.version = newVersion.version;

            Swal.fire({
                title: '回溯成功',
                html: `
                    <div class="text-start">
                        <p class="mb-2">配置已成功回溯到版本 ${version}</p>
                        <p class="mb-0">新版本号：${newVersion.version}</p>
                    </div>
                `,
                icon: 'success'
            });

            // 刷新页面显示
            updateConfigDisplay();
        }
    });
}

// 更新配置显示
function updateConfigDisplay() {
    // 更新版本显示
    document.querySelector('.badge.bg-info').textContent = `当前版本: ${CONFIG.version}`;
    
    // 更新时间显示
    const currentConfig = CONFIG.configHistory[0];
    document.querySelector('.text-muted:nth-child(2)').textContent = `更新时间: ${currentConfig.date}`;
    
    // 更新操作人显示
    document.querySelector('.text-muted:nth-child(4)').textContent = `操作人: ${currentConfig.operator}`;
}

// 查看配置详情
function viewConfigDetail() {
    const currentConfig = CONFIG.configHistory[0].config;
    
    Swal.fire({
        title: '当前配置详情',
        html: `
            <div class="accordion" id="configDetailAccordion">
                <!-- 基本信息 -->
                <div class="accordion-item">
                    <h2 class="accordion-header">
                        <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#basicInfoDetail">
                            基本信息
                        </button>
                    </h2>
                    <div id="basicInfoDetail" class="accordion-collapse collapse show" data-bs-parent="#configDetailAccordion">
                        <div class="accordion-body">
                            <table class="table table-sm">
                                <tr>
                                    <th width="30%">当前版本</th>
                                    <td>${CONFIG.version}</td>
                                </tr>
                                <tr>
                                    <th>最后更新</th>
                                    <td>${CONFIG.configHistory[0].date}</td>
                                </tr>
                                <tr>
                                    <th>操作人</th>
                                    <td>${CONFIG.configHistory[0].operator}</td>
                                </tr>
                                <tr>
                                    <th>版本说明</th>
                                    <td>${CONFIG.configHistory[0].description}</td>
                                </tr>
                            </table>
                        </div>
                    </div>
                </div>

                <!-- 阈值配置 -->
                <div class="accordion-item">
                    <h2 class="accordion-header">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#thresholdDetail">
                            阈值配置
                        </button>
                    </h2>
                    <div id="thresholdDetail" class="accordion-collapse collapse" data-bs-parent="#configDetailAccordion">
                        <div class="accordion-body">
                            <table class="table table-sm">
                                <tr>
                                    <th width="30%">风电产能波动阈值</th>
                                    <td>
                                        <div class="d-flex justify-content-between align-items-center">
                                            <span>${currentConfig.thresholds.windThreshold}%</span>
                                            <span class="badge bg-info">建议值: 15%</span>
                                        </div>
                                    </td>
                                </tr>
                                <tr>
                                    <th>负荷预测偏差阈值</th>
                                    <td>
                                        <div class="d-flex justify-content-between align-items-center">
                                            <span>${currentConfig.thresholds.loadThreshold}%</span>
                                            <span class="badge bg-info">建议值: 10%</span>
                                        </div>
                                    </td>
                                </tr>
                                <tr>
                                    <th>时序预测置信度阈值</th>
                                    <td>
                                        <div class="d-flex justify-content-between align-items-center">
                                            <span>${currentConfig.thresholds.confidenceThreshold}%</span>
                                            <span class="badge bg-info">建议值: 85%</span>
                                        </div>
                                    </td>
                                </tr>
                            </table>
                            <div class="alert alert-info mt-3 mb-0">
                                <i class="fas fa-info-circle me-2"></i>
                                阈值配置影响系统告警的触发条件，请根据实际运行情况适当调整
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 模型参数 -->
                <div class="accordion-item">
                    <h2 class="accordion-header">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#modelDetail">
                            模型参数
                        </button>
                    </h2>
                    <div id="modelDetail" class="accordion-collapse collapse" data-bs-parent="#configDetailAccordion">
                        <div class="accordion-body">
                            <table class="table table-sm">
                                <tr>
                                    <th width="30%">时序窗口大小</th>
                                    <td>
                                        <div class="d-flex justify-content-between align-items-center">
                                            <span>${currentConfig.modelParams.windowSize}</span>
                                            <span class="badge bg-info">建议范围: 24-720</span>
                                        </div>
                                    </td>
                                </tr>
                                <tr>
                                    <th>注意力头数</th>
                                    <td>
                                        <div class="d-flex justify-content-between align-items-center">
                                            <span>${currentConfig.modelParams.attentionHeads}</span>
                                            <span class="badge bg-info">建议范围: 4-16</span>
                                        </div>
                                    </td>
                                </tr>
                                <tr>
                                    <th>学习率</th>
                                    <td>
                                        <div class="d-flex justify-content-between align-items-center">
                                            <span>${currentConfig.modelParams.learningRate}</span>
                                            <span class="badge bg-info">建议范围: 0.0001-0.01</span>
                                        </div>
                                    </td>
                                </tr>
                            </table>
                            <div class="alert alert-warning mt-3 mb-0">
                                <i class="fas fa-exclamation-triangle me-2"></i>
                                模型参数的调整可能会显著影响预测性能，请谨慎修改
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 告警设置 -->
                <div class="accordion-item">
                    <h2 class="accordion-header">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#alertDetail">
                            告警设置
                        </button>
                    </h2>
                    <div id="alertDetail" class="accordion-collapse collapse" data-bs-parent="#configDetailAccordion">
                        <div class="accordion-body">
                            <table class="table table-sm">
                                <tr>
                                    <th width="30%">告警级别</th>
                                    <td>
                                        <span class="badge bg-${getAlertLevelColor(currentConfig.alertSettings.level)}">
                                            ${getAlertLevelName(currentConfig.alertSettings.level)}
                                        </span>
                                    </td>
                                </tr>
                                <tr>
                                    <th>告警方式</th>
                                    <td>
                                        ${currentConfig.alertSettings.methods.email ? '<span class="badge bg-success me-2">邮件通知</span>' : ''}
                                        ${currentConfig.alertSettings.methods.sms ? '<span class="badge bg-success me-2">短信通知</span>' : ''}
                                        ${currentConfig.alertSettings.methods.system ? '<span class="badge bg-success">系统通知</span>' : ''}
                                    </td>
                                </tr>
                                <tr>
                                    <th>告警时间</th>
                                    <td>${getAlertTimeName(currentConfig.alertSettings.time)}</td>
                                </tr>
                            </table>
                            <div class="alert alert-success mt-3 mb-0">
                                <i class="fas fa-check-circle me-2"></i>
                                当前告警配置已启用并正常运行中
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `,
        width: '800px',
        showCloseButton: true,
        showConfirmButton: false
    });
}

// 辅助函数：获取告警级别颜色
function getAlertLevelColor(level) {
    const colors = {
        info: 'info',
        warning: 'warning',
        danger: 'danger'
    };
    return colors[level] || 'secondary';
}

// 辅助函数：获取告警级别名称
function getAlertLevelName(level) {
    const names = {
        info: '提示',
        warning: '警告',
        danger: '严重'
    };
    return names[level] || '未知';
}

// 辅助函数：获取告警时间名称
function getAlertTimeName(time) {
    const names = {
        always: '全天',
        workday: '工作日',
        custom: '自定义'
    };
    return names[time] || time;
}

// 保存单个模型参数
function saveParameter(btn) {
    const input = btn.closest('tr').querySelector('input[data-param-key]');
    const paramKey = input.getAttribute('data-param-key');
    const newValue = input.value;
    const originalValue = input.getAttribute('data-original');

    // Basic validation (can be expanded)
    if (newValue === '') {
        Swal.fire('错误', '参数值不能为空', 'error');
        return;
    }

    // Update the CONFIG object (assuming the first entry is the current config)
    CONFIG.configHistory[0].config.modelParams[paramKey] = newValue;
    input.setAttribute('data-original', newValue); // Update original value after saving

    // Add history entry or update current version description - TBD (for now, just save)

    Swal.fire({
        title: '保存成功',
        text: `参数 ${paramKey} 已更新为 ${newValue}`,
        icon: 'success',
        timer: 1500,
        showConfirmButton: false
    });
}

// 重置单个模型参数
function resetParameter(btn) {
    const input = btn.closest('tr').querySelector('input[data-param-key]');
    const originalValue = input.getAttribute('data-original');
    input.value = originalValue;
    // No need to update CONFIG as it wasn't changed yet

    Swal.fire({
        title: '已重置',
        text: `参数已恢复为 ${originalValue}`,
        icon: 'info',
        timer: 1500,
        showConfirmButton: false
    });
}

// 导出模型参数配置
function exportModelParams() {
    const currentParams = CONFIG.configHistory[0].config.modelParams;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentParams, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", `model_params_${CONFIG.version}.json`);
    document.body.appendChild(downloadAnchorNode); // required for firefox
    downloadAnchorNode.click();
    downloadAnchorNode.remove();

    Swal.fire({
        title: '导出成功',
        text: '模型参数已导出为 JSON 文件',
        icon: 'success',
        timer: 2000,
        showConfirmButton: false
    });
}

// 重置所有模型参数
function resetAllParams() {
    Swal.fire({
        title: '确认重置所有模型参数?',
        text: "此操作会将所有模型参数恢复到当前版本的初始值！",
        icon: 'warning',
        showCancelButton: true,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: '确认重置',
        cancelButtonText: '取消'
    }).then((result) => {
        if (result.isConfirmed) {
            const paramInputs = document.querySelectorAll('#modelParamContent input[data-param-key]');
            paramInputs.forEach(input => {
                const originalValue = input.getAttribute('data-original');
                input.value = originalValue;
                // Optionally, reset the live CONFIG object too if save was immediate
                // const paramKey = input.getAttribute('data-param-key');
                // CONFIG.configHistory[0].config.modelParams[paramKey] = originalValue;
            });

            Swal.fire(
                '已重置!',
                '所有模型参数已恢复.',
                'success'
            );
        }
    });
}