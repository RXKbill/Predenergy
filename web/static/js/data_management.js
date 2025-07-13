document.addEventListener('DOMContentLoaded', function() {
    // REMOVED: No longer needed as tabs were removed from HTML.
    // setTimeout(() => {
    //     console.log('Initializing Bootstrap Tabs (delayed)...');
    //     document.querySelectorAll('.nav-link[data-bs-toggle="tab"]').forEach(function(element) {
    //         bootstrap.Tab.getOrCreateInstance(element);
    //         console.log('Initialized tab for:', element.getAttribute('href'));
    //     });
    // }, 100);

    DataSourceManager.init();
    // 初始化历史数据管理模块
    HistoricalDataManager.init();

    // Add event listener for the Search Input
    const searchInput = document.querySelector('.app-search input');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            DataSourceManager.searchDataSources(e.target.value);
        });
    }
    
    // Add event listener for the Add Data Source Button in modal
    const addDataSourceBtn = document.querySelector('#addDataSourceModal .btn-primary');
    if(addDataSourceBtn && addDataSourceBtn.textContent.includes('添加')) { // Check text to be specific
        addDataSourceBtn.addEventListener('click', addDataSource);
    }

    // Add event listener for the Configure Modal Save Button
    const configConfirmBtn = document.getElementById('configConfirmBtn');
    if (configConfirmBtn) {
        configConfirmBtn.addEventListener('click', saveDataSourceConfig);
    }
});

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    DataSourceManager.init();
});

// 数据源管理模块
const DataSourceManager = {
    constructor() {
        this.init();
    },

    init() {
        // 初始化导航栏点击事件
        document.querySelectorAll('.nav-pills .nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const category = e.target.getAttribute('data-category');
                this.filterDataSources(category);
            });
        });

        // 初始化搜索功能
        const searchInput = document.querySelector('.app-search input');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.searchDataSources(e.target.value);
            });
        }

        // 初始调用一次过滤，显示所有数据源
        this.filterDataSources('all');
    },

    // 存储数据源配置的模板
    configTemplates: {
        modbus: {
            fields: [
                { name: 'host', label: 'IP地址', type: 'text', required: true },
                { name: 'port', label: '端口', type: 'number', required: true },
                { name: 'slave_id', label: '从站ID', type: 'number', required: true },
                { name: 'register_start', label: '起始寄存器', type: 'number', required: true },
                { name: 'register_count', label: '寄存器数量', type: 'number', required: true }
            ]
        },
        opcua: {
            fields: [
                { name: 'endpoint_url', label: '终端URL', type: 'text', required: true },
                { name: 'security_mode', label: '安全模式', type: 'select', options: ['None', 'Sign', 'SignAndEncrypt'] },
                { name: 'node_ids', label: '节点ID列表', type: 'textarea', required: true }
            ]
        },
        weather_api: {
            fields: [
                { name: 'api_key', label: 'API密钥', type: 'password', required: true },
                { name: 'location', label: '位置坐标', type: 'text', required: true },
                { name: 'parameters', label: '气象参数', type: 'select', multiple: true, 
                  options: ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'solar_radiation'] }
            ]
        }
    },

    // 存储所有数据源
    dataSources: [],

    // 初始化
    init() {
        this.bindEvents();
        this.loadDataSources();
    },

    // 绑定事件处理器
    bindEvents() {
        // 数据源类型选择事件
        document.getElementById('dataSourceType').addEventListener('change', (e) => {
            this.updateDynamicConfig(e.target.value);
        });

        // 分类标签点击事件
        document.querySelectorAll('.nav-pills .nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const category = e.target.getAttribute('data-category');
                this.filterDataSources(category);
            });
        });
    },

    // 更新动态配置区域
    updateDynamicConfig(sourceType) {
        const configArea = document.getElementById('dynamicConfigArea');
        const template = this.configTemplates[sourceType];
        
        if (!template) {
            configArea.innerHTML = '<div class="alert alert-info">请选择数据源类型</div>';
            return;
        }

        let html = '';
        template.fields.forEach(field => {
            html += this.generateFieldHtml(field);
        });

        configArea.innerHTML = html;
    },

    // 生成表单字段HTML
    generateFieldHtml(field) {
        let html = `
            <div class="mb-3">
                <label class="form-label">${field.label}</label>
        `;

        switch (field.type) {
            case 'select':
                html += `<select class="form-select" name="${field.name}" ${field.required ? 'required' : ''} ${field.multiple ? 'multiple' : ''}>`;
                field.options.forEach(option => {
                    html += `<option value="${option}">${option}</option>`;
                });
                html += '</select>';
                break;

            case 'textarea':
                html += `<textarea class="form-control" name="${field.name}" rows="3" ${field.required ? 'required' : ''}></textarea>`;
                break;

            default:
                html += `<input type="${field.type}" class="form-control" name="${field.name}" ${field.required ? 'required' : ''}>`;
        }

        html += '</div>';
        return html;
    },

    // 加载数据源列表
    loadDataSources() {
        // 这里应该从后端API获取数据，现在用模拟数据
        this.dataSources = [
            {
                id: 1,
                name: '风机振动传感器',
                type: 'modbus',
                category: 'iot',
                status: 'connected',
                lastUpdate: '2024-02-13 14:30:25',
                config: {
                    protocol: 'Modbus TCP',
                    frequency: '5s/次',
                    parameters: ['振动频率', '转速', '温度']
                }
            },
            {
                id: 2,
                name: 'ECMWF气象数据',
                type: 'weather_api',
                category: 'weather',
                status: 'connected',
                lastUpdate: '2024-02-13 14:00:00',
                config: {
                    source: 'ECMWF API',
                    frequency: '每小时',
                    parameters: ['风速', '风向', '光照强度', '温度']
                }
            },
            {
                id: 3,
                name: '智能电表 - 工厂A',
                type: 'opcua', // 假设是OPC UA
                category: 'energy',
                status: 'connected',
                lastUpdate: '2024-02-13 14:35:00',
                config: {
                    protocol: 'OPC UA',
                    frequency: '15分钟/次',
                    parameters: ['有功功率', '无功功率', '电压', '电流']
                }
            },
            {
                id: 4,
                name: '充电桩群 - 园区B',
                type: 'custom_api', // 自定义API
                category: 'iot',
                status: 'warning', // 状态告警
                lastUpdate: '2024-02-13 13:55:10',
                config: {
                    protocol: 'HTTPS API',
                    frequency: '实时',
                    parameters: ['充电状态', '充电功率', '用户ID', '充电时长']
                }
            },
            {
                id: 5,
                name: '区域电网负荷',
                type: 'database', // 数据库接入
                category: 'energy',
                status: 'disconnected', // 连接断开
                lastUpdate: '2024-02-12 23:00:00',
                config: {
                    source: 'SQL Server',
                    frequency: '每天',
                    parameters: ['总负荷', '峰值负荷', '谷值负荷']
                }
            },
            {
                id: 6,
                name: '能源交易市场报价',
                type: 'websocket', // WebSocket
                category: 'market',
                status: 'connected',
                lastUpdate: '2024-02-13 14:36:15',
                config: {
                    source: '交易平台WS',
                    frequency: '实时推送',
                    parameters: ['买入价', '卖出价', '成交量']
                }
            },
             {
                id: 7,
                name: '光伏逆变器 - 电站C',
                type: 'modbus',
                category: 'iot',
                status: 'connected',
                lastUpdate: '2024-02-13 14:32:00',
                config: {
                    protocol: 'Modbus RTU',
                    frequency: '1分钟/次',
                    parameters: ['直流电压', '直流电流', '交流功率', '转换效率']
                }
            },
            {
                id: 8,
                name: '国家能源政策库',
                type: 'web_crawler', // 网络爬虫
                category: 'policy',
                status: 'connected',
                lastUpdate: '2024-02-10 10:00:00',
                config: {
                    source: '能源局网站',
                    frequency: '每天抓取',
                    parameters: ['政策标题', '发布日期', '政策内容摘要']
                }
            }
        ];

        this.renderDataSources(this.dataSources);
        this.updateOverviewCards();
    },

    // 渲染数据源列表
    renderDataSources(dataSources) {
        const container = document.getElementById('dataSourceList');
        let html = '';

        dataSources.forEach(source => {
            html += this.generateDataSourceCard(source);
        });

        container.innerHTML = html;
        this.initializeCardActions();
    },

    // 生成数据源卡片HTML
    generateDataSourceCard(source) {
        return `
            <div class="col-md-4 mb-4" data-category="${source.category}">
                <div class="card h-100">
                    <div class="card-header">
                        <div class="d-flex justify-content-between align-items-center">
                            <h6 class="mb-0">
                                <span class="connection-status ${source.status}"></span>
                                ${source.name}
                            </h6>
                            <div class="dropdown">
                                <button class="btn btn-sm btn-outline-secondary dropdown-toggle" data-bs-toggle="dropdown">
                                    操作
                                </button>
                                <ul class="dropdown-menu">
                                    <li><a class="dropdown-item" href="#" data-action="configure" data-id="${source.id}">配置参数</a></li>
                                    <li><a class="dropdown-item" href="#" data-action="preview" data-id="${source.id}">数据预览</a></li>
                                    <li><a class="dropdown-item" href="#" data-action="export" data-id="${source.id}">导出数据</a></li>
                                    <li><hr class="dropdown-divider"></li>
                                    <li><a class="dropdown-item text-danger" href="#" data-action="delete" data-id="${source.id}">删除</a></li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    <div class="card-body">
                        <p><strong>类型：</strong>${source.config.protocol || source.type}</p>
                        <p><strong>采集频率：</strong>${source.config.frequency}</p>
                        <p><strong>数据项：</strong>${source.config.parameters.join('、')}</p>
                        <p><strong>最后更新：</strong>${source.lastUpdate}</p>
                        <div class="progress mt-2" style="height: 5px;">
                            <div class="progress-bar bg-success" style="width: ${source.status === 'connected' ? '100%' : '0%'}"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    },

    // 初始化卡片操作
    initializeCardActions() {
        document.querySelectorAll('[data-action]').forEach(element => {
            element.addEventListener('click', (e) => {
                e.preventDefault();
                const action = e.target.getAttribute('data-action');
                const sourceId = e.target.getAttribute('data-id');
                this.handleCardAction(action, sourceId);
            });
        });

        // 添加导出确认按钮的事件监听器
        const exportConfirmBtn = document.getElementById('exportConfirmBtn');
        if (exportConfirmBtn) {
            exportConfirmBtn.addEventListener('click', () => {
                this.confirmExport();
            });
        }
    },

    // 处理卡片操作
    handleCardAction(action, sourceId) {
        switch (action) {
            case 'configure':
                this.showConfigureModal(sourceId);
                break;
            case 'preview':
                this.showPreviewModal(sourceId);
                break;
            case 'export':
                this.exportData(sourceId);
                break;
            case 'delete':
                this.deleteDataSource(sourceId);
                break;
        }
    },

    // 更新概览卡片
    updateOverviewCards() {
        const total = this.dataSources.length;
        const connected = this.dataSources.filter(s => s.status === 'connected').length;
        const warning = this.dataSources.filter(s => s.status === 'warning').length;
        const disconnected = this.dataSources.filter(s => s.status === 'disconnected').length;
        
        // 获取所有卡片
        const cards = document.querySelectorAll('.card');
        
        // 更新已连接数据源卡片
        cards.forEach(card => {
            const titleElement = card.querySelector('.card-title');
            if (!titleElement) return;
            
            const title = titleElement.textContent.trim();
            if (title === '已连接数据源') {
                const countElement = card.querySelector('h2.card-text');
                const progressBar = card.querySelector('.progress-bar');
                if (countElement) countElement.textContent = connected;
                if (progressBar) progressBar.style.width = `${(connected/total)*100}%`;
            }
            else if (title === '数据完整率') {
                const completeness = ((connected + warning) / total * 100).toFixed(1);
                const countElement = card.querySelector('h2.card-text');
                const progressBar = card.querySelector('.progress-bar');
                if (countElement) countElement.textContent = `${completeness}%`;
                if (progressBar) progressBar.style.width = `${completeness}%`;
            }
            else if (title === '异常数据源') {
                const countElement = card.querySelector('h2.card-text');
                const progressBar = card.querySelector('.progress-bar');
                if (countElement) countElement.textContent = disconnected;
                if (progressBar) progressBar.style.width = `${(disconnected/total)*100}%`;
            }
        });
    },

    // 显示配置模态框
    showConfigureModal(sourceId) {
        const source = this.dataSources.find(s => s.id === parseInt(sourceId));
        if (!source) return;

        const modalElement = document.getElementById('configureDataSourceModal');
        const content = document.getElementById('configureContent');
        const saveButton = modalElement.querySelector('.btn-primary');

        saveButton.dataset.sourceId = sourceId;

        // --- Start: Generate Type-Specific Fields ---
        let specificFieldsHtml = '';
        const template = this.configTemplates[source.type];
        if (template) {
            template.fields.forEach(field => {
                // Try to get existing value from source.config, otherwise use empty string
                const currentValue = source.config[field.name] || ''; 
                specificFieldsHtml += this.generateFieldHtmlWithValue(field, currentValue);
            });
        } else {
            specificFieldsHtml = '<div class="alert alert-warning">此类型无特定配置项。</div>';
        }
        // --- End: Generate Type-Specific Fields ---

        let html = `
            <form id="configForm">
                <input type="hidden" name="id" value="${sourceId}">
                <div class="mb-3">
                    <label class="form-label">数据源名称</label>
                    <input type="text" class="form-control" name="name" value="${source.name}" required>
                </div>
                <div class="mb-3">
                    <label class="form-label">数据源类型</label>
                    <input type="text" class="form-control" name="type" value="${source.type}" readonly>
                </div>
                 <div class="mb-3">
                    <label class="form-label">分类</label>
                     <select class="form-select" name="category" required>
                        <option value="iot" ${source.category === 'iot' ? 'selected' : ''}>IoT设备</option>
                        <option value="weather" ${source.category === 'weather' ? 'selected' : ''}>气象数据</option>
                        <option value="energy" ${source.category === 'energy' ? 'selected' : ''}>能耗数据</option>
                        <option value="market" ${source.category === 'market' ? 'selected' : ''}>市场交易</option>
                        <option value="policy" ${source.category === 'policy' ? 'selected' : ''}>政策法规</option>
                    </select>
                </div>
                
                <hr>
                <h5>特定配置</h5>
                <div id="specificConfigArea">
                    ${specificFieldsHtml}
                </div>
                <hr>

                <div class="mb-3">
                    <label class="form-label">采集频率</label>
                    <select class="form-select" name="frequencyValue" required> <!-- Changed name -->
                        <option value="实时" ${source.config.frequency === '实时' ? 'selected' : ''}>实时</option>
                        <option value="5s/次" ${source.config.frequency === '5s/次' ? 'selected' : ''}>每5秒</option>
                        <option value="1分钟/次" ${source.config.frequency === '1分钟/次' ? 'selected' : ''}>每分钟</option>
                        <option value="15分钟/次" ${source.config.frequency === '15分钟/次' ? 'selected' : ''}>每15分钟</option>
                        <option value="每小时" ${source.config.frequency === '每小时' ? 'selected' : ''}>每小时</option>
                        <option value="每天" ${source.config.frequency === '每天' ? 'selected' : ''}>每天</option>
                        <option value="每天抓取" ${source.config.frequency === '每天抓取' ? 'selected' : ''}>每天抓取</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label class="form-label">数据项 (逗号分隔)</label>
                    <input type="text" class="form-control" name="parametersValue" value="${source.config.parameters.join(',')}" required> <!-- Changed name -->
                </div>
                <div class="mb-3">
                    <label class="form-label">连接状态</label>
                    <select class="form-select" name="status" required>
                        <option value="connected" ${source.status === 'connected' ? 'selected' : ''}>已连接</option>
                        <option value="warning" ${source.status === 'warning' ? 'selected' : ''}>告警</option>
                        <option value="disconnected" ${source.status === 'disconnected' ? 'selected' : ''}>已断开</option>
                    </select>
                </div>
            </form>
        `;

        content.innerHTML = html;
        new bootstrap.Modal(modalElement).show();
    },

    // Generate field HTML with a pre-filled value
    generateFieldHtmlWithValue(field, value) {
        let html = `
            <div class="mb-3">
                <label class="form-label">${field.label}</label>
        `;

        switch (field.type) {
            case 'select':
                html += `<select class="form-select" name="${field.name}" ${field.required ? 'required' : ''} ${field.multiple ? 'multiple' : ''}>`;
                field.options.forEach(option => {
                    // Handle single and multiple selects for pre-selection
                    const selected = field.multiple 
                        ? (Array.isArray(value) && value.includes(option) ? 'selected' : '') 
                        : (value === option ? 'selected' : '');
                    html += `<option value="${option}" ${selected}>${option}</option>`;
                });
                html += '</select>';
                break;

            case 'textarea':
                html += `<textarea class="form-control" name="${field.name}" rows="3" ${field.required ? 'required' : ''}>${value || ''}</textarea>`;
                break;
            
            case 'password': // Ensure password fields don't show existing value
                 html += `<input type="password" class="form-control" name="${field.name}" ${field.required ? 'required' : ''} placeholder="(输入以更新)">`;
                 break;

            default: // Handles text, number, etc.
                html += `<input type="${field.type}" class="form-control" name="${field.name}" value="${value || ''}" ${field.required ? 'required' : ''}>`;
        }

        html += '</div>';
        return html;
    },

    // 显示预览模态框
    showPreviewModal(sourceId) {
        const source = this.dataSources.find(s => s.id === parseInt(sourceId));
        if (!source) return;

        const modal = document.getElementById('previewDataModal');
        const content = modal.querySelector('.table-responsive');
        
        // 生成预览数据表格
        let html = `
            <table class="table">
                <thead>
                    <tr>
                        <th>时间戳</th>
                        ${source.config.parameters.map(param => `<th>${param}</th>`).join('')}
                        <th>状态</th>
                    </tr>
                </thead>
                <tbody>
                    ${this.generatePreviewData(source)}
                </tbody>
            </table>
        `;

        content.innerHTML = html;
        new bootstrap.Modal(modal).show();
    },

    // 生成预览数据
    generatePreviewData(source) {
        const now = new Date();
        let html = '';
        
        // 生成最近5条数据
        for (let i = 0; i < 5; i++) {
            const timestamp = new Date(now - i * 60000); // 每分钟一条数据
            const values = source.config.parameters.map(param => {
                // 根据参数类型生成模拟数据
                if (param.includes('温度')) return (20 + Math.random() * 5).toFixed(1);
                if (param.includes('功率')) return (1000 + Math.random() * 500).toFixed(0);
                if (param.includes('电压')) return (220 + Math.random() * 10).toFixed(1);
                if (param.includes('电流')) return (5 + Math.random() * 2).toFixed(1);
                if (param.includes('频率')) return (50 + Math.random() * 0.2).toFixed(1);
                return (Math.random() * 100).toFixed(1);
            });

            html += `
                <tr>
                    <td>${timestamp.toLocaleString()}</td>
                    ${values.map(v => `<td>${v}</td>`).join('')}
                    <td><span class="badge bg-success">正常</span></td>
                </tr>
            `;
        }

        return html;
    },

    // 导出数据
    exportData(sourceId) {
        const source = this.dataSources.find(s => s.id === parseInt(sourceId));
        if (!source) return;

        const modalElement = document.getElementById('exportModal');
        const content = document.getElementById('exportContent');
        const exportButton = modalElement.querySelector('#exportConfirmBtn');

        // Store sourceId on the export button
        exportButton.dataset.sourceId = sourceId; 

        let html = `
            <form id="exportForm">
                <p><strong>导出数据源:</strong> ${source.name}</p>
                <div class="mb-3">
                    <label class="form-label">时间范围</label>
                    <div class="input-group">
                        <input type="datetime-local" class="form-control" name="startTime" required>
                        <span class="input-group-text">至</span>
                        <input type="datetime-local" class="form-control" name="endTime" required>
                    </div>
                </div>
                <div class="mb-3">
                    <label class="form-label">数据项</label>
                    <div class="form-check-scrollable" style="max-height: 150px; overflow-y: auto; border: 1px solid #ced4da; padding: 10px;">
                        ${source.config.parameters.map(param => `
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" name="parameters" value="${param}" checked>
                                <label class="form-check-label">${param}</label>
                            </div>
                        `).join('')}
                    </div>
                </div>
                <div class="mb-3">
                    <label class="form-label">导出格式</label>
                    <select class="form-select" name="format" required>
                        <option value="csv">CSV</option>
                        <option value="excel">Excel</option>
                        <option value="json">JSON</option>
                    </select>
                </div>
            </form>
        `;

        content.innerHTML = html;

        // 初始化模态框
        const modal = new bootstrap.Modal(modalElement);

        // 绑定导出确认按钮事件
        const confirmBtn = modalElement.querySelector('#exportConfirmBtn');
        if (confirmBtn) {
            // 移除旧的事件监听器
            confirmBtn.replaceWith(confirmBtn.cloneNode(true));
            const newConfirmBtn = modalElement.querySelector('#exportConfirmBtn');
            
            // 添加新的事件监听器
            newConfirmBtn.addEventListener('click', () => {
                this.confirmExport();
            });
        }

        // 显示模态框
        modal.show();
    },

    // 删除数据源
    deleteDataSource(sourceId) {
        // 创建确认对话框
        const confirmDialog = document.createElement('div');
        confirmDialog.className = 'modal fade';
        confirmDialog.id = 'deleteConfirmModal';
        confirmDialog.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content bg-dark">
                    <div class="modal-header border-bottom-0">
                        <h5 class="modal-title text-light">删除确认</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body text-light">
                        <p>确定要删除这个数据源吗？此操作不可恢复。</p>
                    </div>
                    <div class="modal-footer border-top-0">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
                        <button type="button" class="btn btn-danger" id="confirmDeleteBtn">删除</button>
                    </div>
                </div>
            </div>
        `;
        
        // 添加对话框到文档
        document.body.appendChild(confirmDialog);
        
        // 初始化模态框
        const modal = new bootstrap.Modal(confirmDialog);
        
        // 显示对话框
        modal.show();
        
        // 处理删除确认
        document.getElementById('confirmDeleteBtn').addEventListener('click', () => {
            // 从数组中移除数据源
            this.dataSources = this.dataSources.filter(s => s.id !== parseInt(sourceId));
            
            // 重新渲染列表
            this.renderDataSources(this.dataSources);
            this.updateOverviewCards();
            
            // 显示成功提示
            this.showNotification('数据源已删除', 'success');
            
            // 关闭并移除对话框
            modal.hide();
            confirmDialog.addEventListener('hidden.bs.modal', () => {
                confirmDialog.remove();
            });
        });
        
        // 对话框关闭时移除元素
        confirmDialog.addEventListener('hidden.bs.modal', () => {
            confirmDialog.remove();
        });
    },

    // 显示通知
    showNotification(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        document.body.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    },

    // 过滤数据源
    filterDataSources(category) {
        // 更新导航项激活状态
        const navLinks = document.querySelectorAll('.nav-pills .nav-link');
        navLinks.forEach(link => {
            if (link.getAttribute('data-category') === category) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });

        // 过滤数据源卡片
        const cards = document.querySelectorAll('#dataSourceList [data-category]');
        cards.forEach(card => {
            if (category === 'all' || card.getAttribute('data-category') === category) {
                card.style.display = 'block';
            } else {
                card.style.display = 'none';
            }
        });
    },

    // 搜索数据源
    searchDataSources(query) {
        const normalizedQuery = query.toLowerCase();
        const filteredSources = this.dataSources.filter(source => 
            source.name.toLowerCase().includes(normalizedQuery) ||
            source.type.toLowerCase().includes(normalizedQuery) ||
            source.category.toLowerCase().includes(normalizedQuery) ||
            source.config.parameters.some(param => param.toLowerCase().includes(normalizedQuery))
        );
        
        this.renderDataSources(filteredSources);
    },

    // 导出数据确认处理
    confirmExport() {
        const modalElement = document.getElementById('exportModal');
        const form = modalElement.querySelector('#exportForm');
        const exportButton = modalElement.querySelector('#exportConfirmBtn');
        const sourceId = parseInt(exportButton.dataset.sourceId);

        if (!form || !sourceId) {
            console.error('无法找到导出表单或数据源ID');
            this.showNotification('导出失败', 'danger');
            return;
        }
        
        if (!form.checkValidity()) {
            form.classList.add('was-validated');
            return;
        }

        const formData = new FormData(form);
        const exportParams = {
            sourceId: sourceId,
            startTime: formData.get('startTime'),
            endTime: formData.get('endTime'),
            parameters: formData.getAll('parameters'),
            format: formData.get('format')
        };

        // 获取数据源信息
        const source = this.dataSources.find(s => s.id === sourceId);
        if (!source) {
            this.showNotification('找不到指定的数据源', 'danger');
            return;
        }

        // 生成模拟数据
        const data = this.generateExportData(source, exportParams);

        // 根据选择的格式导出文件
        switch (exportParams.format) {
            case 'csv':
                this.exportAsCSV(data, source.name);
                break;
            case 'excel':
                this.exportAsExcel(data, source.name);
                break;
            case 'json':
                this.exportAsJSON(data, source.name);
                break;
        }

        // 关闭模态框
        const modal = bootstrap.Modal.getInstance(modalElement);
        modal.hide();

        this.showNotification(`数据已导出为${exportParams.format.toUpperCase()}格式`, 'success');
    },

    // 生成导出数据
    generateExportData(source, params) {
        const data = [];
        const startTime = new Date(params.startTime);
        const endTime = new Date(params.endTime);
        
        // 根据数据源的采集频率确定时间间隔
        let interval;
        switch (source.config.frequency) {
            case '实时':
            case '5s/次':
                interval = 5000; // 5秒
                break;
            case '1分钟/次':
                interval = 60000; // 1分钟
                break;
            case '15分钟/次':
                interval = 900000; // 15分钟
                break;
            case '每小时':
                interval = 3600000; // 1小时
                break;
            case '每天':
            case '每天抓取':
                interval = 86400000; // 24小时
                break;
            default:
                interval = 300000; // 默认5分钟
        }

        // 计算需要生成的数据点数量
        const duration = endTime.getTime() - startTime.getTime();
        const maxPoints = 1000; // 限制最大数据点数，避免文件过大
        const actualInterval = Math.max(interval, Math.floor(duration / maxPoints));

        // 生成时间序列数据
        for (let time = startTime; time <= endTime; time = new Date(time.getTime() + actualInterval)) {
            const row = {
                timestamp: this.formatTimestamp(time, source.config.frequency),
            };

            // 为每个选中的参数生成模拟数据
            params.parameters.forEach(param => {
                row[param] = this.generateParameterValue(param, time);
            });

            data.push(row);
        }

        return data;
    },

    // 格式化时间戳
    formatTimestamp(date, frequency) {
        const pad = (num) => String(num).padStart(2, '0');
        
        const year = date.getFullYear();
        const month = pad(date.getMonth() + 1);
        const day = pad(date.getDate());
        const hours = pad(date.getHours());
        const minutes = pad(date.getMinutes());
        const seconds = pad(date.getSeconds());

        // 根据采集频率返回不同精度的时间格式
        switch (frequency) {
            case '每天':
            case '每天抓取':
                return `${year}-${month}-${day}`;
            case '每小时':
                return `${year}-${month}-${day} ${hours}:00`;
            case '15分钟/次':
                return `${year}-${month}-${day} ${hours}:${Math.floor(minutes / 15) * 15}:00`;
            case '1分钟/次':
                return `${year}-${month}-${day} ${hours}:${minutes}:00`;
            case '实时':
            case '5s/次':
            default:
                return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
        }
    },

    // 生成参数值
    generateParameterValue(param, timestamp) {
        // 基础值 + 周期性变化 + 随机波动
        let baseValue = 0;
        let amplitude = 0;
        let randomRange = 0;

        // 根据参数类型设置基础值和波动范围
        if (param.includes('温度')) {
            baseValue = 25; // 基础温度25℃
            amplitude = 5; // 温度波动±5℃
            randomRange = 0.5; // 随机波动±0.5℃
        } else if (param.includes('功率')) {
            baseValue = 1000; // 基础功率1000W
            amplitude = 500; // 功率波动±500W
            randomRange = 50; // 随机波动±50W
        } else if (param.includes('电压')) {
            baseValue = 220; // 基础电压220V
            amplitude = 10; // 电压波动±10V
            randomRange = 1; // 随机波动±1V
        } else if (param.includes('电流')) {
            baseValue = 5; // 基础电流5A
            amplitude = 2; // 电流波动±2A
            randomRange = 0.2; // 随机波动±0.2A
        } else if (param.includes('频率')) {
            baseValue = 50; // 基础频率50Hz
            amplitude = 0.2; // 频率波动±0.2Hz
            randomRange = 0.05; // 随机波动±0.05Hz
        } else {
            baseValue = 50;
            amplitude = 25;
            randomRange = 5;
        }

        // 生成周期性变化（24小时为周期）
        const hour = timestamp.getHours() + timestamp.getMinutes() / 60;
        const periodicChange = amplitude * Math.sin(hour * Math.PI / 12); // 24小时一个完整周期

        // 添加随机波动
        const randomNoise = (Math.random() - 0.5) * 2 * randomRange;

        // 组合最终值
        let value = baseValue + periodicChange + randomNoise;

        // 确保数值合理（不小于0）
        value = Math.max(0, value);

        // 根据参数类型格式化小数位数
        if (param.includes('温度') || param.includes('电压') || param.includes('电流')) {
            return value.toFixed(1);
        } else if (param.includes('频率')) {
            return value.toFixed(2);
        } else if (param.includes('功率')) {
            return Math.round(value);
        }
        
        return value.toFixed(1);
    },

    // 导出为CSV
    exportAsCSV(data, sourceName) {
        if (!data || data.length === 0) return;

        // 获取表头
        const headers = Object.keys(data[0]);
        
        // 生成CSV内容，添加Excel日期格式识别标记
        let csvContent = '\uFEFF'; // UTF-8 BOM
        
        // 添加表头行
        csvContent += headers.join(',') + '\n';
        
        // 添加数据行
        data.forEach(row => {
            const values = headers.map(header => {
                const value = row[header];
                if (header === 'timestamp') {
                    // 为timestamp添加Excel日期格式标记
                    return `="${value}"`; // Excel会自动识别这种格式为日期
                }
                // 处理包含逗号的值，用引号包裹
                return value.toString().includes(',') ? `"${value}"` : value;
            });
            csvContent += values.join(',') + '\n';
        });

        // 创建Blob对象
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        
        // 生成文件名
        const fileName = `${sourceName}_数据导出_${new Date().toLocaleDateString().replace(/\//g, '-')}.csv`;
        
        // 下载文件
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = fileName;
        link.click();
        URL.revokeObjectURL(link.href);
    },

    // 导出为JSON
    exportAsJSON(data, sourceName) {
        const jsonString = JSON.stringify(data, null, 2);
        const blob = new Blob([jsonString], { type: 'application/json;charset=utf-8;' });
        
        // 生成文件名
        const fileName = `${sourceName}_数据导出_${new Date().toLocaleDateString().replace(/\//g, '-')}.json`;
        
        // 下载文件
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = fileName;
        link.click();
        URL.revokeObjectURL(link.href);
    },

    // 导出为Excel
    exportAsExcel(data, sourceName) {
        // 创建工作簿
        const wb = XLSX.utils.book_new();
        
        // 转换数据为工作表
        const ws = XLSX.utils.json_to_sheet(data);
        
        // 将工作表添加到工作簿
        XLSX.utils.book_append_sheet(wb, ws, '数据导出');
        
        // 生成文件名
        const fileName = `${sourceName}_数据导出_${new Date().toLocaleDateString().replace(/\//g, '-')}.xlsx`;
        
        // 生成Excel文件并下载
        XLSX.writeFile(wb, fileName);
    },
};

// 添加数据源的处理函数
function addDataSource() {
    const form = document.getElementById('addDataSourceForm');
    if (!form.checkValidity()) {
        form.classList.add('was-validated');
        return; 
    }
    const formData = new FormData(form);
    const newSourceData = Object.fromEntries(formData);

    // Find the highest current ID and add 1
    const maxId = DataSourceManager.dataSources.reduce((max, s) => Math.max(max, s.id), 0);
    const newId = maxId + 1;

    // Structure the new source object similarly to mock data
    const newSource = {
        id: newId,
        name: newSourceData.name,
        type: newSourceData.type,
        category: 'iot', // Default or determine based on type
        status: 'connected', // Default status
        lastUpdate: new Date().toLocaleString(),
        config: { 
            protocol: newSourceData.type, // Simplified assumption
            frequency: 'N/A', // Get from dynamic fields if added
            parameters: [] // Get from dynamic fields if added
            // TODO: Populate specific config based on dynamic fields
        }
    };

    // Add to the manager's list
    DataSourceManager.dataSources.push(newSource);

    // Refresh UI
    DataSourceManager.renderDataSources(DataSourceManager.dataSources);
    DataSourceManager.updateOverviewCards();
    DataSourceManager.showNotification(`数据源 '${newSource.name}' 添加成功`, 'success');

    // Close modal and reset form
    const modal = bootstrap.Modal.getInstance(document.getElementById('addDataSourceModal'));
    modal.hide();
    form.reset();
    form.classList.remove('was-validated');
    document.getElementById('dynamicConfigArea').innerHTML = '<div class="alert alert-info">请选择数据源类型</div>'; // Reset dynamic area
}

// 保存数据源配置的处理函数
function saveDataSourceConfig() {
    const modalElement = document.getElementById('configureDataSourceModal');
    const form = modalElement.querySelector('#configForm');
    const saveButton = modalElement.querySelector('.btn-primary');
    const sourceId = parseInt(saveButton.dataset.sourceId);

    if (!form || !sourceId) {
        console.error('无法找到配置表单或数据源ID');
        DataSourceManager.showNotification('保存配置失败', 'danger');
        return;
    }

    if (!form.checkValidity()) {
        form.classList.add('was-validated');
        return;
    }

    const formData = new FormData(form);
    const configData = {}; // Use a plain object to gather all form data
    formData.forEach((value, key) => {
        // Handle potential multiple values for checkboxes/multi-selects if needed later
        configData[key] = value; 
    });
    
    const sourceIndex = DataSourceManager.dataSources.findIndex(s => s.id === sourceId);

    if (sourceIndex === -1) {
        console.error('无法找到要更新的数据源:', sourceId);
        DataSourceManager.showNotification('保存配置失败', 'danger');
        return;
    }

    // Update the data source object
    const sourceToUpdate = DataSourceManager.dataSources[sourceIndex];
    sourceToUpdate.name = configData.name;
    // Type is readonly, no need to update: sourceToUpdate.type = configData.type;
    sourceToUpdate.category = configData.category;
    sourceToUpdate.status = configData.status;
    sourceToUpdate.config.frequency = configData.frequencyValue; // Use updated name
    sourceToUpdate.config.parameters = configData.parametersValue.split(',').map(p => p.trim()).filter(p => p); // Use updated name
    sourceToUpdate.lastUpdate = new Date().toLocaleString();

    // --- Start: Update specific config fields ---
    const template = DataSourceManager.configTemplates[sourceToUpdate.type];
    if (template) {
        template.fields.forEach(field => {
            if (configData.hasOwnProperty(field.name)) {
                 // Don't save empty password fields unless intended
                 if (field.type === 'password' && !configData[field.name]) {
                     // Skip saving password if input is empty
                 } else {
                    sourceToUpdate.config[field.name] = configData[field.name];
                 }
            }
        });
    }
    // --- End: Update specific config fields ---

    // Refresh UI
    DataSourceManager.renderDataSources(DataSourceManager.dataSources);
    DataSourceManager.updateOverviewCards();
    DataSourceManager.showNotification(`数据源 '${configData.name}' 配置已更新`, 'success');

    // Close modal
    const modal = bootstrap.Modal.getInstance(modalElement);
    modal.hide();
}

// 历史数据管理模块
const HistoricalDataManager = {
    // 存储历史数据
    historicalData: {},
    
    // 站点/设备数据
    stations: {
        wind: [
            { id: 'wind1', name: '张北风电场' },
            { id: 'wind2', name: '大同风电场' },
            { id: 'wind3', name: '河北沿海风电场' }
        ],
        solar: [
            { id: 'solar1', name: '青海光伏电站' },
            { id: 'solar2', name: '宁夏沙漠光伏基地' },
            { id: 'solar3', name: '山西屋顶光伏' }
        ],
        storage: [
            { id: 'storage1', name: '庐山抽水蓄能电站' },
            { id: 'storage2', name: '张家口储能电站' }
        ],
        charging: [
            { id: 'charging1', name: '北京海淀充电站' },
            { id: 'charging2', name: '上海浦东充电桩群' },
            { id: 'charging3', name: '广州番禺充电中心' }
        ]
    },
    
    // 数据字段映射
    dataFields: {
        generation: ['发电量', '功率', '等效利用小时数', '容量因子'],
        price: ['现货价格', '日前价格', '辅助服务价格', '碳价格'],
        load: ['用电负荷', '峰值负荷', '谷值负荷', '负荷率'],
        charging: ['充电量', '充电时长', '充电功率', '充电费用']
    },
    
    // 敏感数据区域
    sensitiveAreas: ['wind2', 'solar2', 'storage1'],
    
    // 初始化
    init() {
        console.log('初始化历史数据管理模块');
        
        // 确保所有必要的DOM元素都存在
        const requiredElements = [
            'sceneType',
            'dataType',
            'startDate',
            'endDate',
            'granularity',
            'stationSelector',
            'filterField',
            'queryHistoryBtn',
            'exportHistoryBtn',
            'viewModeChartBtn',
            'viewModeTableBtn',
            'includeQualityData'
        ];

        const missingElements = requiredElements.filter(id => !document.getElementById(id));
        if (missingElements.length > 0) {
            console.error('缺少必要的DOM元素:', missingElements);
            return;
        }

        // 绑定事件
        this.bindEvents();
        
        // 加载示例数据
        this.loadSampleData();
        
        // 初始化站点选择器
        this.populateStationSelector();
        
        // 设置默认日期范围（最近30天）
        const today = new Date();
        const thirtyDaysAgo = new Date();
        thirtyDaysAgo.setDate(today.getDate() - 30);
        
        const endDateInput = document.getElementById('endDate');
        const startDateInput = document.getElementById('startDate');
        endDateInput.valueAsDate = today;
        startDateInput.valueAsDate = thirtyDaysAgo;
        
        // 设置默认选项
        const sceneTypeSelect = document.getElementById('sceneType');
        const dataTypeSelect = document.getElementById('dataType');
        
        // 默认选择风电场景和发电记录
        sceneTypeSelect.value = 'wind';
        dataTypeSelect.value = 'generation';
        
        // 触发场景类型变更事件以更新站点选择器
        this.handleSceneTypeChange({ target: sceneTypeSelect });
        
        // 加载初始数据
        this.loadInitialData();
        
        console.log('历史数据加载完成，可用数据键:', Object.keys(this.historicalData));
        
        // 初始化高级过滤条件
        this.initializeAdvancedFilters();
    },
    
    // 初始化高级过滤条件
    initializeAdvancedFilters() {
        // 获取相关DOM元素
        const dataTypeSelect = document.getElementById('dataType');
        const filterField = document.getElementById('filterField');
        const filterOperator = document.getElementById('filterOperator');
        const filterValue = document.getElementById('filterValue');
        const filterValueHelp = document.getElementById('filterValueHelp');
        const resetFilterBtn = document.getElementById('resetFilterBtn');
        
        // 监听数据类型变更
        dataTypeSelect.addEventListener('change', (e) => {
            console.log('数据类型变更:', e.target.value);
            this.populateFilterFields(e.target.value);
        });
        
        // 监听过滤字段变更
        filterField.addEventListener('change', (e) => {
            console.log('过滤字段变更:', e.target.value);
            this.handleFilterFieldChange();
        });
        
        // 监听操作符变更
        filterOperator.addEventListener('change', (e) => {
            console.log('操作符变更:', e.target.value);
            this.updateFilterValueInput();
        });
        
        // 监听过滤值输入
        filterValue.addEventListener('input', (e) => {
            console.log('过滤值输入:', e.target.value);
            this.validateFilterValue();
        });
        
        // 监听重置按钮点击
        resetFilterBtn.addEventListener('click', () => {
            console.log('重置过滤条件');
            this.resetFilters();
        });
        
        // 初始化工具提示
        const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
        tooltips.forEach(tooltip => {
            new bootstrap.Tooltip(tooltip);
        });
        
        // 初始化数据类型的过滤字段
        if (dataTypeSelect.value) {
            this.populateFilterFields(dataTypeSelect.value);
        }
    },
    
    // 更新过滤值输入框
    updateFilterValueInput() {
        const filterField = document.getElementById('filterField');
        const filterOperator = document.getElementById('filterOperator');
        const filterValue = document.getElementById('filterValue');
        const filterValueHelp = document.getElementById('filterValueHelp');
        
        // 获取选中字段的配置
        const fields = JSON.parse(filterField.dataset.fields || '[]');
        const selectedField = fields.find(f => f.field === filterField.value);
        
        if (!selectedField) {
            filterValue.type = 'text';
            filterValue.placeholder = '输入过滤值';
            filterValueHelp.textContent = '';
            return;
        }
        
        // 根据操作符更新输入框
        const operator = filterOperator.value;
        if (operator === 'between') {
            filterValue.type = 'text';
            filterValue.placeholder = '最小值,最大值';
            filterValueHelp.textContent = `请输入两个值，用逗号分隔，范围：${selectedField.min} - ${selectedField.max}`;
        } else {
            filterValue.type = selectedField.type === 'number' ? 'number' : 'text';
            filterValue.placeholder = `输入${selectedField.field}`;
            if (selectedField.type === 'number') {
                filterValueHelp.textContent = `取值范围：${selectedField.min} - ${selectedField.max} ${selectedField.unit}`;
            } else {
                filterValueHelp.textContent = '';
            }
        }
        
        // 更新输入框属性
        if (selectedField.type === 'number' && operator !== 'between') {
            filterValue.min = selectedField.min;
            filterValue.max = selectedField.max;
            filterValue.step = '0.01';
        } else {
            filterValue.removeAttribute('min');
            filterValue.removeAttribute('max');
            filterValue.removeAttribute('step');
        }
    },
    
    // 重置过滤条件
    resetFilters() {
        const form = document.getElementById('historyDataQueryForm');
        form.reset();
        
        // 重置过滤字段
        this.resetFilterValueInput();
        
        // 清除验证状态
        const filterValue = document.getElementById('filterValue');
        filterValue.setCustomValidity('');
        filterValue.classList.remove('is-invalid', 'is-valid');
        
        // 重置帮助文本
        document.getElementById('filterValueHelp').textContent = '';
        
        // 触发数据类型变更事件以重新加载过滤字段
        const dataType = document.getElementById('dataType');
        this.populateFilterFields(dataType.value);
    },
    
    // 验证过滤值
    validateFilterValue() {
        const filterField = document.getElementById('filterField');
        const filterOperator = document.getElementById('filterOperator');
        const filterValue = document.getElementById('filterValue');
        
        // 获取字段配置
        const fields = JSON.parse(filterField.dataset.fields || '[]');
        const selectedField = fields.find(f => f.field === filterField.value);
        
        if (!selectedField) return;
        
        const value = filterValue.value;
        let isValid = true;
        let errorMessage = '';
        
        if (filterOperator.value === 'between') {
            // 验证区间值
            const [min, max] = value.split(',').map(v => parseFloat(v.trim()));
            if (value.split(',').length !== 2 || isNaN(min) || isNaN(max)) {
                isValid = false;
                errorMessage = '请输入两个有效的数字，用逗号分隔';
            } else if (min >= max) {
                isValid = false;
                errorMessage = '最小值必须小于最大值';
            } else if (min < selectedField.min || max > selectedField.max) {
                isValid = false;
                errorMessage = `取值范围必须在 ${selectedField.min} - ${selectedField.max} 之间`;
            }
        } else if (selectedField.type === 'number') {
            // 验证数值
            const numValue = parseFloat(value);
            if (isNaN(numValue)) {
                isValid = false;
                errorMessage = '请输入有效的数字';
            } else if (numValue < selectedField.min || numValue > selectedField.max) {
                isValid = false;
                errorMessage = `取值范围必须在 ${selectedField.min} - ${selectedField.max} 之间`;
            }
        }
        
        // 更新验证状态
        if (isValid) {
            filterValue.setCustomValidity('');
            filterValue.classList.remove('is-invalid');
            filterValue.classList.add('is-valid');
        } else {
            filterValue.setCustomValidity(errorMessage);
            filterValue.classList.remove('is-valid');
            filterValue.classList.add('is-invalid');
        }
        
        filterValue.reportValidity();
    },
    
    // 加载初始数据
    loadInitialData() {
        console.log('加载初始数据');
        
        // 构造初始查询参数
        const initialParams = {
            sceneType: 'wind',
            dataType: 'generation',
            startDate: document.getElementById('startDate').value,
            endDate: document.getElementById('endDate').value,
            granularity: document.getElementById('granularity').value,
            station: '',  // 不指定具体站点，显示所有站点数据
            includeQuality: false
        };

        console.log('初始查询参数:', initialParams);
        
        // 查询数据
        const results = this.queryHistoricalData(initialParams);
        console.log('初始查询结果:', results);
        
        // 显示结果
        if (results && results.data && results.data.length > 0) {
            console.log('查询到初始数据，显示结果，数据量:', results.data.length);
            document.getElementById('historyResultArea').style.display = 'block';
            document.getElementById('noDataMessage').style.display = 'none';
            this.displayResults(results, initialParams);
            
            // 默认显示图表视图
            this.switchViewMode('chart');
        } else {
            console.log('未查询到初始数据，显示无数据提示');
            document.getElementById('historyResultArea').style.display = 'none';
            document.getElementById('noDataMessage').style.display = 'block';
        }
    },
    
    // 绑定事件
    bindEvents() {
        console.log('绑定事件处理器');
        
        // 场景类型变更事件
        const sceneTypeSelect = document.getElementById('sceneType');
        sceneTypeSelect.addEventListener('change', (e) => {
            console.log('场景类型变更:', e.target.value);
            this.handleSceneTypeChange(e);
        });
        
        // 数据类型变更事件
        const dataTypeSelect = document.getElementById('dataType');
        dataTypeSelect.addEventListener('change', (e) => {
            console.log('数据类型变更:', e.target.value);
            this.handleDataTypeChange(e);
        });
        
        // 查询按钮点击事件
        const queryBtn = document.getElementById('queryHistoryBtn');
        queryBtn.addEventListener('click', () => {
            console.log('点击查询按钮');
            this.handleQuerySubmit();
        });
        
        // 导出按钮点击事件
        const exportBtn = document.getElementById('exportHistoryBtn');
        exportBtn.addEventListener('click', () => {
            console.log('点击导出按钮');
            this.handleExportData();
        });
        
        // 表格/图表切换事件
        const chartBtn = document.getElementById('viewModeChartBtn');
        const tableBtn = document.getElementById('viewModeTableBtn');
        chartBtn.addEventListener('click', () => this.switchViewMode('chart'));
        tableBtn.addEventListener('click', () => this.switchViewMode('table'));
        
        // 数据质量指标复选框事件
        const qualityCheckbox = document.getElementById('includeQualityData');
        qualityCheckbox.addEventListener('change', (e) => this.toggleQualitySection(e));

        // 重置按钮点击事件
        const resetBtn = document.getElementById('resetFilterBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                // 清空表单后隐藏结果区域
                document.getElementById('historyResultArea').style.display = 'none';
                document.getElementById('noDataMessage').style.display = 'none';
                document.getElementById('dataQualitySection').style.display = 'none';
            });
        }
        
        console.log('事件处理器绑定完成');
    },
    
    // 验证日期
    validateDates() {
        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;

        if (!startDate || !endDate) {
            return false;
        }

        const start = new Date(startDate);
        const end = new Date(endDate);

        if (start > end) {
            this.showNotification('开始日期不能晚于结束日期', 'warning');
            return false;
        }

        return true;
    },
    
    // 处理查询提交
    handleQuerySubmit() {
        // 获取表单数据
        const form = document.getElementById('historyDataQueryForm');
        if (!form) return;
        
        const formData = new FormData(form);
        const params = {
            sceneType: formData.get('sceneType'),
            dataType: formData.get('dataType'),
            startDate: formData.get('startDate'),
            endDate: formData.get('endDate'),
            granularity: formData.get('granularity'),
            station: formData.get('station'),
            includeQuality: formData.get('includeQualityData') === 'on',
            filterField: formData.get('filterField'),
            filterOperator: formData.get('filterOperator'),
            filterValue: formData.get('filterValue')
        };
        
        // 验证必填字段
        if (!params.sceneType || !params.dataType) {
            this.showNotification('请选择场景类型和数据类型', 'warning');
            return;
        }
        
        // 验证日期
        if (!params.startDate || !params.endDate) {
            this.showNotification('请选择时间范围', 'warning');
            return;
        }

        // 验证日期范围
        if (!this.validateDates()) {
            return;
        }

        // 应用过滤条件
        let results = this.queryHistoricalData(params);
        if (params.filterField && params.filterOperator && params.filterValue) {
            results.data = this.applyFilters(results.data, params);
        }
        
        if (results && results.data && results.data.length > 0) {
            document.getElementById('historyResultArea').style.display = 'block';
            document.getElementById('noDataMessage').style.display = 'none';
            this.displayResults(results, params);

            // 如果选中了数据质量指标，更新质量图表
            if (params.includeQuality) {
                document.getElementById('dataQualitySection').style.display = 'block';
                this.updateQualityCharts(results.quality);
            } else {
                document.getElementById('dataQualitySection').style.display = 'none';
            }
        } else {
            document.getElementById('historyResultArea').style.display = 'none';
            document.getElementById('noDataMessage').style.display = 'block';
            document.getElementById('dataQualitySection').style.display = 'none';
        }
    },
    
    // 处理场景类型变更
    handleSceneTypeChange(e) {
        const sceneType = e.target.value;
        console.log('场景类型变更:', sceneType);
        this.populateStationSelector(sceneType);

        // 如果选择了场景类型，自动刷新数据显示
        if (sceneType) {
            const dataType = document.getElementById('dataType').value;
            if (dataType) {
                const params = {
                    sceneType: sceneType,
                    dataType: dataType,
                    startDate: document.getElementById('startDate').value,
                    endDate: document.getElementById('endDate').value,
                    granularity: document.getElementById('granularity').value,
                    station: document.getElementById('stationSelector').value,
                    includeQuality: document.getElementById('includeQualityData').checked
                };
                
                const results = this.queryHistoricalData(params);
                if (results && results.data && results.data.length > 0) {
                    document.getElementById('historyResultArea').style.display = 'block';
                    document.getElementById('noDataMessage').style.display = 'none';
                    this.displayResults(results, params);
                } else {
                    document.getElementById('historyResultArea').style.display = 'none';
                    document.getElementById('noDataMessage').style.display = 'block';
                }
            }
        }
    },
    
    // 处理数据类型变更
    handleDataTypeChange(e) {
        const dataType = e.target.value;
        this.populateFilterFields(dataType);
    },
    
    // 填充站点选择器
    populateStationSelector(sceneType = '') {
        const selector = document.getElementById('stationSelector');
        selector.innerHTML = '<option value="">请选择站点/设备</option>';
        
        if (!sceneType) return;
        
        const stations = this.stations[sceneType] || [];
        stations.forEach(station => {
            const option = document.createElement('option');
            option.value = station.id;
            option.textContent = station.name;
            selector.appendChild(option);
        });
    },
    
    // 填充过滤字段
    populateFilterFields(dataType = '') {
        const selector = document.getElementById('filterField');
        selector.innerHTML = '<option value="">选择字段</option>';
        
        if (!dataType) return;
        
        // 根据数据类型定义可过滤字段及其属性
        const filterFields = {
            generation: [
                { field: '发电量', unit: 'MWh', type: 'number', min: 0, max: 10000 },
                { field: '功率', unit: 'MW', type: 'number', min: 0, max: 5000 },
                { field: '等效利用小时数', unit: 'h', type: 'number', min: 0, max: 8760 },
                { field: '容量因子', unit: '%', type: 'number', min: 0, max: 100 }
            ],
            price: [
                { field: '现货价格', unit: '元/MWh', type: 'number', min: 0, max: 2000 },
                { field: '日前价格', unit: '元/MWh', type: 'number', min: 0, max: 2000 },
                { field: '辅助服务价格', unit: '元/MWh', type: 'number', min: 0, max: 500 },
                { field: '碳价格', unit: '元/吨', type: 'number', min: 0, max: 200 }
            ],
            load: [
                { field: '用电负荷', unit: 'MW', type: 'number', min: 0, max: 10000 },
                { field: '峰值负荷', unit: 'MW', type: 'number', min: 0, max: 12000 },
                { field: '谷值负荷', unit: 'MW', type: 'number', min: 0, max: 8000 },
                { field: '负荷率', unit: '%', type: 'number', min: 0, max: 100 }
            ],
            charging: [
                { field: '充电量', unit: 'kWh', type: 'number', min: 0, max: 1000 },
                { field: '充电时长', unit: 'h', type: 'number', min: 0, max: 24 },
                { field: '充电功率', unit: 'kW', type: 'number', min: 0, max: 500 },
                { field: '充电费用', unit: '元', type: 'number', min: 0, max: 1000 }
            ]
        };
        
        const fields = filterFields[dataType] || [];
        
        // 存储字段配置到选择器的dataset中
        selector.dataset.fields = JSON.stringify(fields);
        
        // 清空现有选项
        selector.innerHTML = '<option value="">选择字段</option>';
        
        // 添加新选项
        fields.forEach(field => {
            const option = document.createElement('option');
            option.value = field.field;
            option.textContent = `${field.field} (${field.unit})`;
            selector.appendChild(option);
        });
        
        // 重置过滤值输入框
        this.resetFilterValueInput();
        
        // 更新操作符选项
        this.updateOperatorOptions();
    },
    
    // 更新操作符选项
    updateOperatorOptions() {
        const filterField = document.getElementById('filterField');
        const filterOperator = document.getElementById('filterOperator');
        const filterValue = document.getElementById('filterValue');
        
        // 获取选中字段的配置
        const fields = JSON.parse(filterField.dataset.fields || '[]');
        const selectedField = fields.find(f => f.field === filterField.value);
        
        // 清空现有选项
        filterOperator.innerHTML = '';
        
        if (!selectedField) {
            // 默认操作符
            const defaultOperators = [
                { value: '=', label: '等于 (=)' }
            ];
            defaultOperators.forEach(op => {
                const option = document.createElement('option');
                option.value = op.value;
                option.textContent = op.label;
                filterOperator.appendChild(option);
            });
            return;
        }
        
        // 根据字段类型获取操作符
        const operators = this.getOperatorsForType(selectedField.type);
        
        // 添加操作符选项
        operators.forEach(op => {
            const option = document.createElement('option');
            option.value = op.value;
            option.textContent = op.label;
            filterOperator.appendChild(option);
        });
        
        // 更新过滤值输入框
        this.updateFilterValueInput();
    },
    
    // 处理过滤字段变更
    handleFilterFieldChange() {
        const filterField = document.getElementById('filterField');
        const filterOperator = document.getElementById('filterOperator');
        filterValue.type = selectedField.type === 'number' ? 'number' : 'text';
        if (selectedField.type === 'number') {
            filterValue.min = selectedField.min;
            filterValue.max = selectedField.max;
            filterValue.step = '0.01';
        } else {
            filterValue.removeAttribute('min');
            filterValue.removeAttribute('max');
            filterValue.removeAttribute('step');
        }
        filterValue.placeholder = `输入${selectedField.field} (${selectedField.unit})`;
        
        // 添加输入验证
        filterValue.addEventListener('input', () => this.validateFilterValue(selectedField));
    },
    
    // 获取字段类型对应的操作符
    getOperatorsForType(type) {
        const operators = {
            number: [
                { value: '=', label: '等于 (=)' },
                { value: '>', label: '大于 (>)' },
                { value: '<', label: '小于 (<)' },
                { value: '>=', label: '大于等于 (>=)' },
                { value: '<=', label: '小于等于 (<=)' },
                { value: 'between', label: '介于 (between)' }
            ],
            text: [
                { value: '=', label: '等于 (=)' },
                { value: 'contains', label: '包含' },
                { value: 'startsWith', label: '开头是' },
                { value: 'endsWith', label: '结尾是' }
            ]
        };
        return operators[type] || operators.text;
    },
    
    // 验证过滤值
    validateFilterValue(fieldConfig) {
        const filterValue = document.getElementById('filterValue');
        const value = filterValue.value;
        
        if (fieldConfig.type === 'number') {
            const numValue = parseFloat(value);
            if (isNaN(numValue)) {
                filterValue.setCustomValidity('请输入有效的数字');
            } else if (numValue < fieldConfig.min || numValue > fieldConfig.max) {
                filterValue.setCustomValidity(`请输入 ${fieldConfig.min} 到 ${fieldConfig.max} 之间的值`);
            } else {
                filterValue.setCustomValidity('');
            }
        }
        
        filterValue.reportValidity();
    },
    
    // 应用过滤条件
    applyFilters(data, filterParams) {
        if (!filterParams.filterField || !filterParams.filterOperator || !filterParams.filterValue) {
            return data;
        }
        
        const fields = JSON.parse(document.getElementById('filterField').dataset.fields || '[]');
        const fieldConfig = fields.find(f => f.field === filterParams.filterField);
        
        if (!fieldConfig) return data;
        
        return data.filter(item => {
            const itemValue = item[filterParams.filterField];
            const filterValue = filterParams.filterValue;
            
            if (fieldConfig.type === 'number') {
                const numItemValue = parseFloat(itemValue);
                const numFilterValue = parseFloat(filterValue);
                
                switch (filterParams.filterOperator) {
                    case '=': return numItemValue === numFilterValue;
                    case '>': return numItemValue > numFilterValue;
                    case '<': return numItemValue < numFilterValue;
                    case '>=': return numItemValue >= numFilterValue;
                    case '<=': return numItemValue <= numFilterValue;
                    case 'between':
                        const [min, max] = filterValue.split(',').map(v => parseFloat(v.trim()));
                        return numItemValue >= min && numItemValue <= max;
                    default: return true;
                }
            } else {
                const strItemValue = String(itemValue).toLowerCase();
                const strFilterValue = String(filterValue).toLowerCase();
                
                switch (filterParams.filterOperator) {
                    case '=': return strItemValue === strFilterValue;
                    case 'contains': return strItemValue.includes(strFilterValue);
                    case 'startsWith': return strItemValue.startsWith(strFilterValue);
                    case 'endsWith': return strItemValue.endsWith(strFilterValue);
                    default: return true;
                }
            }
        });
    },
    
    // 更新查询历史数据函数
    queryHistoricalData(params) {
        // 在实际应用中，这里应该调用后端API获取数据
        // 当前使用预先加载的示例数据模拟
        console.log('查询历史数据，参数:', params);
        
        // 构造查询键
        let key;
        switch(params.dataType) {
            case 'generation':
                key = `${params.sceneType}_generation`;
                break;
            case 'price':
                key = `${params.sceneType}_price`;
                break;
            case 'load':
                key = `${params.sceneType}_load`;
                break;
            case 'charging':
                key = `${params.sceneType}_charging`;
                break;
            default:
                console.error('未知的数据类型:', params.dataType);
                return null;
        }
        
        let data = this.historicalData[key] || [];
        
        // 筛选日期范围
        const startDate = new Date(params.startDate);
        const endDate = new Date(params.endDate);
        data = data.filter(item => {
            const itemDate = new Date(item.date);
            return itemDate >= startDate && itemDate <= endDate;
        });
        
        // 应用站点筛选
        if (params.station) {
            data = data.filter(item => item.stationId === params.station);
        }
        
        // 应用高级过滤条件
        if (params.filterField && params.filterOperator && params.filterValue) {
            data = this.applyFilters(data, params);
        }
        
        // 按照粒度对数据进行聚合
        data = this.aggregateByGranularity(data, params.granularity);
        
        // 计算数据质量指标
        const quality = this.calculateQualityMetrics(data);
        
        return {
            data: data,
            quality: quality,
            total: data.length,
            timeSpan: this.calculateTimeSpan(startDate, endDate)
        };
    },
    
    // 按粒度聚合数据
    aggregateByGranularity(data, granularity) {
        // 简单实现，实际应用中需要更复杂的聚合逻辑
        return data;
    },
    
    // 计算时间跨度（天）
    calculateTimeSpan(startDate, endDate) {
        const diffTime = Math.abs(endDate - startDate);
        return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    },
    
    // 计算数据质量指标
    calculateQualityMetrics(data) {
        // 在实际应用中，这些计算应在后端进行
        // 这里使用模拟数据
        return {
            missingBefore: 2.3,
            missingAfter: 0,
            anomalyBefore: 1.5,
            anomalyAfter: 0.8,
            duplicateBefore: 0.5,
            duplicateAfter: 0,
            consistencyBefore: 97.2,
            consistencyAfter: 99.5
        };
    },
    
    // 显示查询结果
    displayResults(results, params) {
        // 更新结果摘要
        document.getElementById('totalRecords').textContent = results.total;
        document.getElementById('timeSpan').textContent = results.timeSpan;
        
        const missing = 2.3; // 模拟缺失率
        const anomaly = 1.5; // 模拟异常率
        
        document.getElementById('missingRate').textContent = `${missing}%`;
        document.getElementById('missingRateBar').style.width = `${missing}%`;
        document.getElementById('anomalyRate').textContent = `${anomaly}%`;
        document.getElementById('anomalyRateBar').style.width = `${anomaly}%`;
        
        // 渲染图表
        this.renderChart(results.data, params);
        
        // 渲染表格
        this.renderTable(results.data, params);
    },
    
    // 渲染图表
    renderChart(data, params) {
        console.log('渲染图表，数据点数量:', data.length);
        const chartContainer = document.getElementById('historyDataChart');
        
        // 创建图表容器，使用响应式布局
        chartContainer.innerHTML = `
            <div style="width:100%; height:400px; position:relative;">
                <canvas id="simpleChart" style="width:100%; height:100%;"></canvas>
            </div>
            <div class="chart-legend mt-3">
                <div class="d-flex justify-content-center align-items-center flex-wrap">
                    <div class="me-4">
                        <span class="badge bg-primary me-2">●</span>
                        <span>正常数据</span>
                    </div>
                    <div class="me-4">
                        <span class="badge bg-danger me-2">●</span>
                        <span>异常数据</span>
                    </div>
                    <div class="me-4">
                        <span class="text-muted">数据类型: ${this.getDataTypeLabel(params.dataType)}</span>
                    </div>
                    <div class="me-4">
                        <span class="text-muted">站点: ${params.station ? this.getStationName(params.sceneType, params.station) : '全部'}</span>
                    </div>
                    <div>
                        <span class="text-muted">时间粒度: ${this.getGranularityLabel(params.granularity)}</span>
                    </div>
                </div>
            </div>
        `;

        // 获取容器宽度并设置canvas尺寸
        const canvas = document.getElementById('simpleChart');
        const container = canvas.parentElement;
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;

        // 按日期排序
        const sortedData = [...data].sort((a, b) => new Date(a.date) - new Date(b.date));
        
        // 获取日期和值
        const dates = sortedData.map(item => item.date);
        const values = sortedData.map(item => item.value);
        const statuses = sortedData.map(item => item.status);
        
        // 获取单位
        const unit = data[0]?.unit || '';
        
        // 绘制图表
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
        // 找到最大值和最小值，并添加合适的边界
            const maxValue = Math.max(...values);
            const minValue = Math.min(...values);
        const valueRange = maxValue - minValue;
        const paddedMax = maxValue + valueRange * 0.1;
        const paddedMin = Math.max(0, minValue - valueRange * 0.1); // 确保最小值不小于0
        const range = paddedMax - paddedMin;
            
            // 设置边距
        const padding = {
            left: Math.round(canvas.width * 0.12),    // 12% 左边距
            right: Math.round(canvas.width * 0.05),   // 5% 右边距
            top: Math.round(canvas.height * 0.1),     // 10% 顶部边距
            bottom: Math.round(canvas.height * 0.15)  // 15% 底部边距
        };
        
        const chartWidth = canvas.width - padding.left - padding.right;
        const chartHeight = canvas.height - padding.top - padding.bottom;
        
        // 绘制标题
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        const title = `${this.getSceneTypeLabel(params.sceneType)} - ${this.getDataTypeLabel(params.dataType)}`;
        ctx.fillText(title, canvas.width / 2, padding.top - 10);
        
        // 绘制Y轴
            ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, canvas.height - padding.bottom);
            ctx.strokeStyle = '#666';
            ctx.lineWidth = 1;
            ctx.stroke();
        
        // 绘制X轴
        ctx.beginPath();
        ctx.moveTo(padding.left, canvas.height - padding.bottom);
        ctx.lineTo(canvas.width - padding.right, canvas.height - padding.bottom);
        ctx.stroke();
        
        // 绘制Y轴刻度和网格线
        ctx.fillStyle = '#fff';
        ctx.font = '12px Arial';
        ctx.textAlign = 'right';
        
        const ySteps = 5;
        for (let i = 0; i <= ySteps; i++) {
            const y = padding.top + (i / ySteps) * chartHeight;
            const value = paddedMax - (i / ySteps) * range;
            
            // 绘制刻度线
            ctx.beginPath();
            ctx.moveTo(padding.left - 5, y);
            ctx.lineTo(padding.left, y);
            ctx.stroke();
            
            // 绘制网格线
            ctx.beginPath();
            ctx.strokeStyle = '#1a2634';
            ctx.setLineDash([2, 2]);
            ctx.moveTo(padding.left, y);
            ctx.lineTo(canvas.width - padding.right, y);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.strokeStyle = '#666';
            
            // 绘制刻度值
            ctx.fillText(this.formatValue(value) + ' ' + unit, padding.left - 10, y + 4);
        }
            
            // 绘制数据点和连线
        if (values.length > 0) {
            // 绘制连线
            ctx.beginPath();
            values.forEach((value, index) => {
                const x = padding.left + (index / (values.length - 1)) * chartWidth;
                const yRatio = (paddedMax - value) / range;
                const y = padding.top + yRatio * chartHeight;
                
                if (index === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            ctx.strokeStyle = '#3498db';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // 绘制数据点
            values.forEach((value, index) => {
                const x = padding.left + (index / (values.length - 1)) * chartWidth;
                const yRatio = (paddedMax - value) / range;
                const y = padding.top + yRatio * chartHeight;
                
                    ctx.beginPath();
                ctx.fillStyle = statuses[index] === 'normal' ? '#3498db' : '#e74c3c';
                    ctx.arc(x, y, 4, 0, Math.PI * 2);
                    ctx.fill();
                
                // 在异常点上添加标记
                if (statuses[index] === 'anomaly') {
                    ctx.beginPath();
                    ctx.strokeStyle = '#e74c3c';
                    ctx.lineWidth = 1;
                    const markSize = 8;
                    ctx.moveTo(x - markSize, y - markSize);
                    ctx.lineTo(x + markSize, y + markSize);
                    ctx.moveTo(x + markSize, y - markSize);
                    ctx.lineTo(x - markSize, y + markSize);
                    ctx.stroke();
                }
                
                // 添加数据点悬停效果
                this.addHoverEffect(canvas, x, y, value, dates[index], unit);
            });
        }
        
        // 绘制X轴刻度
        ctx.fillStyle = '#fff';
            ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        
        // 计算合适的日期标签间隔
        const maxLabels = Math.floor(chartWidth / 100);
        const dateStep = Math.max(1, Math.ceil(dates.length / maxLabels));
        
            for (let i = 0; i < dates.length; i += dateStep) {
            const x = padding.left + (i / (dates.length - 1)) * chartWidth;
                const date = new Date(dates[i]);
            const dateStr = this.formatDate(date, params.granularity);
            
            // 绘制刻度线
            ctx.beginPath();
            ctx.strokeStyle = '#666';
            ctx.moveTo(x, canvas.height - padding.bottom);
            ctx.lineTo(x, canvas.height - padding.bottom + 5);
            ctx.stroke();
            
            // 绘制日期标签
            ctx.save();
            ctx.translate(x, canvas.height - padding.bottom + 20);
            ctx.rotate(-Math.PI / 6); // 倾斜30度
            ctx.fillText(dateStr, 0, 0);
            ctx.restore();
        }
    },
    
    // 格式化数值
    formatValue(value) {
        if (value >= 1000) {
            return (value / 1000).toFixed(1) + 'k';
        }
        return value.toFixed(1);
    },
    
    // 格式化日期
    formatDate(date, granularity) {
        const month = date.getMonth() + 1;
        const day = date.getDate();
        const hours = date.getHours();
        const minutes = date.getMinutes();
        
        switch (granularity) {
            case '5min':
            case '15min':
                return `${month}/${day} ${hours}:${minutes.toString().padStart(2, '0')}`;
            case 'hour':
                return `${month}/${day} ${hours}:00`;
            case 'month':
                return `${month}月`;
            default: // day
                return `${month}/${day}`;
        }
    },
    
    // 获取站点名称
    getStationName(sceneType, stationId) {
        const station = this.stations[sceneType]?.find(s => s.id === stationId);
        return station ? station.name : stationId;
    },
    
    // 获取时间粒度标签
    getGranularityLabel(granularity) {
        const labels = {
            '5min': '5分钟',
            '15min': '15分钟',
            'hour': '1小时',
            'day': '1天',
            'month': '1月'
        };
        return labels[granularity] || granularity;
    },
    
    // 添加悬停效果
    addHoverEffect(canvas, x, y, value, date, unit) {
        let isHovering = false;
        const radius = 10;
        
        canvas.addEventListener('mousemove', (event) => {
            const rect = canvas.getBoundingClientRect();
            const mouseX = event.clientX - rect.left;
            const mouseY = event.clientY - rect.top;
            
            const distance = Math.sqrt(Math.pow(mouseX - x, 2) + Math.pow(mouseY - y, 2));
            
            if (distance <= radius) {
                if (!isHovering) {
                    isHovering = true;
                    canvas.style.cursor = 'pointer';
                    
                    // 显示提示框
                    const tooltip = document.createElement('div');
                    tooltip.className = 'chart-tooltip';
                    tooltip.style.position = 'absolute';
                    tooltip.style.left = (event.clientX + 10) + 'px';
                    tooltip.style.top = (event.clientY - 10) + 'px';
                    tooltip.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
                    tooltip.style.color = '#fff';
                    tooltip.style.padding = '5px 10px';
                    tooltip.style.borderRadius = '4px';
                    tooltip.style.fontSize = '12px';
                    tooltip.style.zIndex = '1000';
                    tooltip.innerHTML = `
                        日期: ${date}<br>
                        数值: ${value} ${unit}
                    `;
                    document.body.appendChild(tooltip);
                    
                    // 移除旧的提示框
                    canvas.addEventListener('mouseout', () => {
                        isHovering = false;
                        canvas.style.cursor = 'default';
                        tooltip.remove();
                    }, { once: true });
                }
            } else if (isHovering) {
                isHovering = false;
                canvas.style.cursor = 'default';
                const tooltip = document.querySelector('.chart-tooltip');
                if (tooltip) {
                    tooltip.remove();
                }
            }
        });
    },
    
    // 获取场景类型标签
    getSceneTypeLabel(sceneType) {
        const labels = {
            'wind': '风电场景',
            'solar': '光伏场景',
            'storage': '储能场景',
            'charging': '充电场景'
        };
        return labels[sceneType] || sceneType;
    },
    
    // 获取数据类型标签
    getDataTypeLabel(dataType) {
        const labels = {
            'generation': '发电记录',
            'price': '交易价格',
            'load': '负荷曲线',
            'charging': '充电记录'
        };
        return labels[dataType] || dataType;
    },
    
    // 渲染表格
    renderTable(data, params) {
        console.log('渲染表格，数据点数量:', data.length);
        const tableBody = document.querySelector('#dataResultTable tbody');
        tableBody.innerHTML = '';
        
        // 按日期排序
        const sortedData = [...data].sort((a, b) => new Date(b.date) - new Date(a.date));
        
        // 分页参数
        const pageSize = 10;
        const totalPages = Math.ceil(sortedData.length / pageSize);
        
        // 从 URL 获取当前页码，如果没有则默认为1
        const urlParams = new URLSearchParams(window.location.search);
        let currentPage = parseInt(urlParams.get('page')) || 1;
        
        // 确保页码在有效范围内
        currentPage = Math.max(1, Math.min(currentPage, totalPages));
        
        // 计算当前页的数据范围
        const startIndex = (currentPage - 1) * pageSize;
        const endIndex = Math.min(startIndex + pageSize, sortedData.length);
        const displayData = sortedData.slice(startIndex, endIndex);
        
        if (displayData.length === 0) {
            const emptyRow = document.createElement('tr');
            emptyRow.innerHTML = `<td colspan="4" class="text-center">未找到符合条件的数据</td>`;
            tableBody.appendChild(emptyRow);
        } else {
            displayData.forEach(item => {
                const row = document.createElement('tr');
                
                const dateCell = document.createElement('td');
                dateCell.textContent = item.date;
                row.appendChild(dateCell);
                
                const valueCell = document.createElement('td');
                valueCell.textContent = item.value;
                row.appendChild(valueCell);
                
                const unitCell = document.createElement('td');
                unitCell.textContent = item.unit;
                row.appendChild(unitCell);
                
                const statusCell = document.createElement('td');
                const statusBadge = document.createElement('span');
                statusBadge.className = `badge bg-${item.status === 'normal' ? 'success' : 'warning'}`;
                statusBadge.textContent = item.status === 'normal' ? '正常' : '异常';
                statusCell.appendChild(statusBadge);
                row.appendChild(statusCell);
                
                tableBody.appendChild(row);
            });
        }
        
        // 更新分页信息
        document.getElementById('startRecord').textContent = sortedData.length > 0 ? startIndex + 1 : 0;
        document.getElementById('endRecord').textContent = endIndex;
        document.getElementById('totalRecordsTable').textContent = sortedData.length;
        
        // 生成分页控件
        this.renderPagination(sortedData.length, pageSize, currentPage);
    },
    
    // 渲染分页控件
    renderPagination(totalItems, pageSize, currentPage) {
        const pagination = document.getElementById('tablePagination');
        pagination.innerHTML = '';
        
        const totalPages = Math.ceil(totalItems / pageSize);
        
        // 如果只有一页，不显示分页
        if (totalPages <= 1) return;
        
        // 创建分页项目
        const createPageItem = (page, text, isActive = false, isDisabled = false) => {
            const li = document.createElement('li');
            li.className = `page-item ${isActive ? 'active' : ''} ${isDisabled ? 'disabled' : ''}`;
            
            const a = document.createElement('a');
            a.className = 'page-link';
            a.href = '#';
            a.innerHTML = text;
            
            if (!isDisabled) {
                a.addEventListener('click', (e) => {
                    e.preventDefault();
                    // 使用箭头函数保持 this 上下文
                    this.handlePageChange(page);
                });
            }
            
            li.appendChild(a);
            return li;
        };
        
        // 上一页按钮
        pagination.appendChild(createPageItem(
            currentPage - 1,
            '&laquo;',
            false,
            currentPage === 1
        ));
        
        // 计算要显示的页码范围
        let startPage = Math.max(1, currentPage - 2);
        let endPage = Math.min(totalPages, startPage + 4);
        
        // 调整起始页，确保始终显示5个页码（如果有）
        if (endPage - startPage < 4) {
            startPage = Math.max(1, endPage - 4);
        }
        
        // 第一页
        if (startPage > 1) {
            pagination.appendChild(createPageItem(1, '1'));
            if (startPage > 2) {
                pagination.appendChild(createPageItem(null, '...', false, true));
            }
        }
        
        // 页码按钮
        for (let i = startPage; i <= endPage; i++) {
            pagination.appendChild(createPageItem(i, i.toString(), i === currentPage));
        }
        
        // 最后一页
        if (endPage < totalPages) {
            if (endPage < totalPages - 1) {
                pagination.appendChild(createPageItem(null, '...', false, true));
            }
            pagination.appendChild(createPageItem(totalPages, totalPages.toString()));
        }
        
        // 下一页按钮
        pagination.appendChild(createPageItem(
            currentPage + 1,
            '&raquo;',
            false,
            currentPage === totalPages
        ));
    },
    
    // 处理页码变更
    handlePageChange(page) {
        if (!page) return; // 忽略省略号点击
        
        // 更新URL参数
        const url = new URL(window.location.href);
        url.searchParams.set('page', page);
        window.history.pushState({}, '', url);
        
        // 获取当前数据并重新渲染表格
        const currentData = this.getCurrentData();
        if (currentData && currentData.data) {
            this.renderTable(currentData.data, currentData.params);
        }
    },
    
    // 切换视图模式（表格/图表）
    switchViewMode(mode) {
        const chartView = document.getElementById('historyDataChart');
        const tableView = document.getElementById('historyDataTable');
        const chartBtn = document.getElementById('viewModeChartBtn');
        const tableBtn = document.getElementById('viewModeTableBtn');
        
        if (!chartView || !tableView || !chartBtn || !tableBtn) {
            console.error('找不到必要的DOM元素');
            return;
        }
        
        // 重置所有按钮样式，保持基础类
        chartBtn.className = 'btn btn-sm';
        tableBtn.className = 'btn btn-sm';
        
        if (mode === 'chart') {
            // 显示图表视图
            chartView.style.display = 'block';
            tableView.style.display = 'none';
            // 更新按钮样式
            chartBtn.classList.add('btn-primary');
            tableBtn.classList.add('btn-light');
            // 如果有数据，重新渲染图表以确保正确显示
            const currentData = this.getCurrentData();
            if (currentData && currentData.data && currentData.data.length > 0) {
                this.renderChart(currentData.data, currentData.params);
            }
        } else {
            // 显示表格视图
            chartView.style.display = 'none';
            tableView.style.display = 'block';
            // 更新按钮样式
            chartBtn.classList.add('btn-light');
            tableBtn.classList.add('btn-primary');
            // 如果有数据，重新渲染表格以确保正确显示
            const currentData = this.getCurrentData();
            if (currentData && currentData.data && currentData.data.length > 0) {
                this.renderTable(currentData.data, currentData.params);
            }
        }
    },
    
    // 切换数据质量部分显示
    toggleQualitySection(e) {
        const checked = e.target.checked;
        document.getElementById('dataQualitySection').style.display = checked && document.getElementById('historyResultArea').style.display !== 'none' ? 'block' : 'none';
    },
    
    // 更新数据质量图表
    updateQualityCharts(qualityData) {
        // 更新质量指标数据
        for (const [key, value] of Object.entries(qualityData)) {
            const element = document.getElementById(key);
            if (element) {
                element.textContent = typeof value === 'number' ? `${value}%` : value;
            }
        }
        
        // 绘制完整性对比图表
        const completenessChart = document.getElementById('completenessChart');
        completenessChart.innerHTML = '<canvas id="completenessCanvas" style="width:100%; height:280px;"></canvas>';
        const completenessCtx = document.getElementById('completenessCanvas').getContext('2d');
        
        // 设置画布尺寸
        const canvas = completenessCtx.canvas;
        canvas.width = completenessChart.clientWidth;
        canvas.height = 280;
        
        // 绘制完整性对比柱状图
        this.drawCompletenessChart(completenessCtx, qualityData);
        
        // 绘制异常分布图表
        const anomalyChart = document.getElementById('anomalyChart');
        anomalyChart.innerHTML = '<canvas id="anomalyCanvas" style="width:100%; height:280px;"></canvas>';
        const anomalyCtx = document.getElementById('anomalyCanvas').getContext('2d');
        
        // 设置画布尺寸
        const anomalyCanvas = anomalyCtx.canvas;
        anomalyCanvas.width = anomalyChart.clientWidth;
        anomalyCanvas.height = 280;
        
        // 绘制异常分布饼图
        this.drawAnomalyChart(anomalyCtx, qualityData);
    },
    
    // 绘制完整性对比柱状图
    drawCompletenessChart(ctx, data) {
        const canvas = ctx.canvas;
        const width = canvas.width;
        const height = canvas.height;
        
        // 设置边距
        const padding = {
            left: 60,
            right: 30,
            top: 30,
            bottom: 50
        };
        
        // 清空画布
        ctx.clearRect(0, 0, width, height);
        
        // 绘制标题
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('数据完整性对比', width / 2, padding.top - 10);
        
        // 定义数据
        const categories = ['缺失值', '异常值', '重复值'];
        const beforeValues = [data.missingBefore, data.anomalyBefore, data.duplicateBefore];
        const afterValues = [data.missingAfter, data.anomalyAfter, data.duplicateAfter];
        
        // 计算最大值以确定比例
        const maxValue = Math.max(...beforeValues, ...afterValues) * 1.2;
        
        // 计算柱状图参数
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;
        const barCount = categories.length;
        const groupWidth = chartWidth / barCount;
        const barWidth = groupWidth * 0.35;
        const barGap = groupWidth * 0.1;
        
        // 绘制Y轴
        ctx.beginPath();
        ctx.strokeStyle = '#666';
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, height - padding.bottom);
        ctx.stroke();
        
        // 绘制X轴
        ctx.beginPath();
        ctx.moveTo(padding.left, height - padding.bottom);
        ctx.lineTo(width - padding.right, height - padding.bottom);
        ctx.stroke();
        
        // 绘制Y轴刻度和网格线
        const ySteps = 5;
        ctx.textAlign = 'right';
        ctx.font = '12px Arial';
        
        for (let i = 0; i <= ySteps; i++) {
            const value = maxValue * (1 - i / ySteps);
            const y = padding.top + (i / ySteps) * chartHeight;
            
            // 绘制刻度线
            ctx.beginPath();
            ctx.moveTo(padding.left - 5, y);
            ctx.lineTo(padding.left, y);
            ctx.stroke();
            
            // 绘制网格线
            ctx.beginPath();
            ctx.strokeStyle = '#1a2634';
            ctx.setLineDash([2, 2]);
            ctx.moveTo(padding.left, y);
            ctx.lineTo(width - padding.right, y);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.strokeStyle = '#666';
            
            // 绘制刻度值
            ctx.fillStyle = '#fff';
            ctx.fillText(value.toFixed(1) + '%', padding.left - 10, y + 4);
        }
        
        // 绘制柱状图
        categories.forEach((category, index) => {
            const x = padding.left + groupWidth * index + groupWidth / 2 - barWidth - barGap / 2;
            
            // 绘制处理前的柱
            const beforeHeight = (beforeValues[index] / maxValue) * chartHeight;
            ctx.fillStyle = '#e74c3c';
            ctx.fillRect(x, height - padding.bottom - beforeHeight, barWidth, beforeHeight);
            
            // 绘制处理后的柱
            const afterHeight = (afterValues[index] / maxValue) * chartHeight;
            ctx.fillStyle = '#2ecc71';
            ctx.fillRect(x + barWidth + barGap, height - padding.bottom - afterHeight, barWidth, afterHeight);
            
            // 绘制类别标签
            ctx.fillStyle = '#fff';
            ctx.textAlign = 'center';
            ctx.font = '12px Arial';
            ctx.fillText(category, x + barWidth + barGap / 2, height - padding.bottom + 20);
        });
        
        // 绘制图例
        const legendY = height - padding.bottom + 40;
        ctx.textAlign = 'left';
        ctx.font = '12px Arial';
        
        // 处理前图例
        ctx.fillStyle = '#e74c3c';
        ctx.fillRect(padding.left, legendY, 15, 15);
        ctx.fillStyle = '#fff';
        ctx.fillText('处理前', padding.left + 20, legendY + 12);
        
        // 处理后图例
        ctx.fillStyle = '#2ecc71';
        ctx.fillRect(padding.left + 100, legendY, 15, 15);
        ctx.fillStyle = '#fff';
        ctx.fillText('处理后', padding.left + 120, legendY + 12);
    },
    
    // 绘制异常分布饼图
    drawAnomalyChart(ctx, data) {
        const canvas = ctx.canvas;
        const width = canvas.width;
        const height = canvas.height;
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) / 3;
        
        // 清空画布
        ctx.clearRect(0, 0, width, height);
        
        // 绘制标题
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('数据异常分布', width / 2, 30);
        
        // 定义异常类型数据
        const anomalyTypes = [
            { type: '缺失值', value: data.missingAfter, color: '#3498db' },
            { type: '异常值', value: data.anomalyAfter, color: '#e74c3c' },
            { type: '重复值', value: data.duplicateAfter, color: '#f1c40f' },
            { type: '正常值', value: 100 - data.missingAfter - data.anomalyAfter - data.duplicateAfter, color: '#2ecc71' }
        ];
        
        // 计算总和
        const total = anomalyTypes.reduce((sum, item) => sum + item.value, 0);
        
        // 绘制饼图
        let startAngle = -Math.PI / 2; // 从12点钟方向开始
        
        anomalyTypes.forEach((item, index) => {
            const sliceAngle = (item.value / total) * Math.PI * 2;
            
            // 绘制扇形
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.arc(centerX, centerY, radius, startAngle, startAngle + sliceAngle);
            ctx.closePath();
            ctx.fillStyle = item.color;
            ctx.fill();
            
            // 计算标签位置
            const labelAngle = startAngle + sliceAngle / 2;
            const labelRadius = radius * 1.3;
            const labelX = centerX + Math.cos(labelAngle) * labelRadius;
            const labelY = centerY + Math.sin(labelAngle) * labelRadius;
            
            // 绘制标签
            ctx.fillStyle = '#fff';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(`${item.type} (${item.value.toFixed(1)}%)`, labelX, labelY);
            
            // 绘制连接线
            const lineStartX = centerX + Math.cos(labelAngle) * radius;
            const lineStartY = centerY + Math.sin(labelAngle) * radius;
            
            ctx.beginPath();
            ctx.strokeStyle = '#666';
            ctx.moveTo(lineStartX, lineStartY);
            ctx.lineTo(labelX, labelY);
            ctx.stroke();
            
            startAngle += sliceAngle;
        });
        
        // 绘制中心圆环
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius * 0.6, 0, Math.PI * 2);
        ctx.fillStyle = '#2c3e50';
        ctx.fill();
        
        // 绘制中心文字
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('数据质量', centerX, centerY - 10);
        ctx.font = '14px Arial';
        ctx.fillText(`${data.consistencyAfter.toFixed(1)}%`, centerX, centerY + 15);
    },
    
    // 处理数据导出
    handleExportData() {
        // 获取当前查询表单数据
        const form = document.getElementById('historyDataQueryForm');
        if (!form) {
            this.showNotification('无法找到查询表单', 'danger');
            return;
        }

        const formData = new FormData(form);
        const queryParams = {
            sceneType: formData.get('sceneType'),
            dataType: formData.get('dataType'),
            startDate: formData.get('startDate'),
            endDate: formData.get('endDate'),
            granularity: formData.get('granularity'),
            station: formData.get('station')
        };
        
        // 验证必填字段
        if (!queryParams.sceneType || !queryParams.dataType) {
            this.showNotification('请选择场景类型和数据类型', 'warning');
            return;
        }
        
        // 验证日期
        if (!queryParams.startDate || !queryParams.endDate) {
            this.showNotification('请选择时间范围', 'warning');
            return;
        }

        // 查询数据
        const results = this.queryHistoricalData(queryParams);
        
        if (!results || !results.data || results.data.length === 0) {
            this.showNotification('没有符合条件的数据可导出', 'warning');
            return;
        }

        // 创建导出选项对话框
        const exportDialog = document.createElement('div');
        exportDialog.className = 'modal fade';
        exportDialog.id = 'historyExportDialog';
        exportDialog.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content bg-dark">
                    <div class="modal-header">
                        <h5 class="modal-title text-light">导出选项</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <form id="historyExportForm">
                            <div class="mb-3">
                                <label class="form-label text-light">导出格式</label>
                                <select class="form-select bg-dark text-light" id="exportFormat">
                                    <option value="csv">CSV</option>
                                    <option value="excel">Excel</option>
                                    <option value="json">JSON</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <label class="form-label text-light">数值精度</label>
                                <select class="form-select bg-dark text-light" id="precision">
                                    <option value="0">整数</option>
                                    <option value="1">1位小数</option>
                                    <option value="2" selected>2位小数</option>
                                    <option value="3">3位小数</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="includeMetadata" checked>
                                    <label class="form-check-label text-light">包含元数据</label>
                                </div>
                            </div>
                            <div class="mb-3">
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="includeQuality" checked>
                                    <label class="form-check-label text-light">包含数据质量信息</label>
                                </div>
                            </div>
                        </form>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
                        <button type="button" class="btn btn-primary" id="confirmHistoryExport">确认导出</button>
                    </div>
                </div>
            </div>
        `;
        
        // 添加对话框到文档
        document.body.appendChild(exportDialog);
        
        // 初始化Bootstrap模态框
        const modal = new bootstrap.Modal(exportDialog);
        
        // 绑定导出确认事件
        const confirmBtn = exportDialog.querySelector('#confirmHistoryExport');
        confirmBtn.addEventListener('click', () => {
            const format = document.getElementById('exportFormat').value;
            const precision = parseInt(document.getElementById('precision').value);
            const includeMetadata = document.getElementById('includeMetadata').checked;
            const includeQuality = document.getElementById('includeQuality').checked;
            
            // 准备导出数据
            const exportData = results.data.map(item => ({
                日期: item.date,
                数值: Number(item.value).toFixed(precision),
                单位: item.unit,
                状态: item.status === 'normal' ? '正常' : '异常'
            }));
            
            // 如果包含数据质量信息，添加相关字段
            if (includeQuality) {
                exportData.forEach(item => {
                    item.质量标记 = this.getQualityMark(item);
                });
            }
            
            // 准备元数据
            const metadata = includeMetadata ? {
                导出时间: new Date().toLocaleString(),
                场景类型: this.getSceneTypeLabel(queryParams.sceneType),
                数据类型: this.getDataTypeLabel(queryParams.dataType),
                时间范围: `${queryParams.startDate} 至 ${queryParams.endDate}`,
                数据粒度: this.getGranularityLabel(queryParams.granularity),
                站点: queryParams.station ? this.getStationName(queryParams.sceneType, queryParams.station) : '全部',
                数据质量: {
                    总记录数: results.total,
                    缺失率: results.quality.missingAfter + '%',
                    异常率: results.quality.anomalyAfter + '%',
                    完整性: results.quality.consistencyAfter + '%'
                }
            } : null;
            
            // 生成文件名
            const fileName = `历史数据_${this.getSceneTypeLabel(queryParams.sceneType)}_${this.getDataTypeLabel(queryParams.dataType)}_${new Date().toLocaleDateString().replace(/\//g, '-')}`;
            
            // 根据选择的格式导出文件
            switch (format) {
                case 'csv':
                    this.exportHistoryToCSV(exportData, metadata, fileName);
                    break;
                case 'excel':
                    this.exportHistoryToExcel(exportData, metadata, fileName);
                    break;
                case 'json':
                    this.exportHistoryToJSON(exportData, metadata, fileName);
                    break;
            }
            
            // 关闭对话框
            modal.hide();
            exportDialog.addEventListener('hidden.bs.modal', () => {
                exportDialog.remove();
            });
            
            // 显示成功提示
            this.showNotification(`数据已导出为${format.toUpperCase()}格式`, 'success');
        });
        
        // 显示对话框
        modal.show();
        
        // 对话框关闭时移除元素
        exportDialog.addEventListener('hidden.bs.modal', () => {
            exportDialog.remove();
        });
    },

    // 导出历史数据为CSV
    exportHistoryToCSV(data, metadata, fileName) {
        let csvContent = '\uFEFF'; // UTF-8 BOM
        
        // 添加元数据
        if (metadata) {
            for (const [key, value] of Object.entries(metadata)) {
                if (typeof value === 'object') {
                    csvContent += `${key}:\n`;
                    for (const [subKey, subValue] of Object.entries(value)) {
                        csvContent += `${subKey},${subValue}\n`;
                    }
                } else {
                    csvContent += `${key},${value}\n`;
                }
            }
            csvContent += '\n'; // 添加空行分隔
        }
        
        // 添加表头
        const headers = Object.keys(data[0]);
        csvContent += headers.join(',') + '\n';
        
        // 添加数据行
        data.forEach(row => {
            const values = headers.map(header => {
                const value = row[header];
                return value.toString().includes(',') ? `"${value}"` : value;
            });
            csvContent += values.join(',') + '\n';
        });
        
        // 创建并下载文件
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `${fileName}.csv`;
        link.click();
        URL.revokeObjectURL(link.href);
    },

    // 导出历史数据为Excel
    exportHistoryToExcel(data, metadata, fileName) {
        // 创建工作簿
        const wb = XLSX.utils.book_new();
        
        // 如果包含元数据，创建元数据工作表
        if (metadata) {
            const metadataArray = [];
            for (const [key, value] of Object.entries(metadata)) {
                if (typeof value === 'object') {
                    metadataArray.push([key + ':']);
                    for (const [subKey, subValue] of Object.entries(value)) {
                        metadataArray.push([subKey, subValue]);
                    }
                } else {
                    metadataArray.push([key, value]);
                }
            }
            const wsMetadata = XLSX.utils.aoa_to_sheet(metadataArray);
            XLSX.utils.book_append_sheet(wb, wsMetadata, '元数据');
        }
        
        // 创建数据工作表
        const ws = XLSX.utils.json_to_sheet(data);
        XLSX.utils.book_append_sheet(wb, ws, '数据');
        
        // 导出文件
        XLSX.writeFile(wb, `${fileName}.xlsx`);
    },

    // 导出历史数据为JSON
    exportHistoryToJSON(data, metadata, fileName) {
        const exportData = {
            metadata: metadata,
            data: data
        };
        
        const jsonString = JSON.stringify(exportData, null, 2);
        const blob = new Blob([jsonString], { type: 'application/json;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `${fileName}.json`;
        link.click();
        URL.revokeObjectURL(link.href);
    },

    // 显示通知
    showNotification(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        document.body.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    },
    
    // 加载示例数据
    loadSampleData() {
        console.log('开始加载示例数据');
        
        // 获取当前日期，用于生成包含当前时间的数据
        const today = new Date();
        const currentYear = today.getFullYear();
        const startYear = currentYear - 1; // 生成从去年到明年的数据
        const endYear = currentYear + 1;
        
        // 生成数据的起止时间
        const startDate = `${startYear}-01-01`;
        const endDate = `${endYear}-12-31`;
        
        // 为每个风电场生成数据
        const windStations = ['wind1', 'wind2', 'wind3'];
        const solarStations = ['solar1', 'solar2', 'solar3'];
        const storageStations = ['storage1', 'storage2'];
        const chargingStations = ['charging1', 'charging2', 'charging3'];
        
        this.historicalData = {};
        
        // 生成风电场数据
        windStations.forEach(stationId => {
            const stationData = {
                generation: this.generateSampleData(
                    startDate,
                    endDate,
                    stationId,
            '风电场日发电量',
            'MWh',
            [1000, 3000],
            0.1
                ),
                price: this.generateSampleData(
                    startDate,
                    endDate,
                    stationId,
                    '风电交易价格',
                    '元/MWh',
                    [300, 600],
                    0.05
                ),
                load: this.generateSampleData(
                    startDate,
                    endDate,
                    stationId,
                    '风电负荷',
                    'MW',
                    [800, 2500],
                    0.08
                )
            };
            
            // 合并到主数据集
            this.historicalData.wind_generation = (this.historicalData.wind_generation || []).concat(stationData.generation);
            this.historicalData.wind_price = (this.historicalData.wind_price || []).concat(stationData.price);
            this.historicalData.wind_load = (this.historicalData.wind_load || []).concat(stationData.load);
        });
        
        // 生成光伏电站数据
        solarStations.forEach(stationId => {
            const stationData = {
                generation: this.generateSampleData(
                    startDate,
                    endDate,
                    stationId,
            '光伏电站日发电量',
            'MWh',
            [500, 2000],
            0.05
                ),
                price: this.generateSampleData(
                    startDate,
                    endDate,
                    stationId,
                    '光伏交易价格',
                    '元/MWh',
                    [280, 550],
                    0.05
                ),
                load: this.generateSampleData(
                    startDate,
                    endDate,
                    stationId,
                    '光伏负荷',
                    'MW',
                    [400, 1800],
                    0.08
                )
            };
            
            this.historicalData.solar_generation = (this.historicalData.solar_generation || []).concat(stationData.generation);
            this.historicalData.solar_price = (this.historicalData.solar_price || []).concat(stationData.price);
            this.historicalData.solar_load = (this.historicalData.solar_load || []).concat(stationData.load);
        });
        
        // 生成储能电站数据
        storageStations.forEach(stationId => {
            const stationData = {
                generation: this.generateSampleData(
                    startDate,
                    endDate,
                    stationId,
                    '储能放电量',
                    'MWh',
                    [200, 1000],
                    0.05
                ),
                price: this.generateSampleData(
                    startDate,
                    endDate,
                    stationId,
                    '储能价格',
                    '元/MWh',
                    [350, 800],
                    0.1
                ),
                load: this.generateSampleData(
                    startDate,
                    endDate,
                    stationId,
                    '储能负荷',
            'MW',
                    [100, 800],
                    0.08
                )
            };
            
            this.historicalData.storage_generation = (this.historicalData.storage_generation || []).concat(stationData.generation);
            this.historicalData.storage_price = (this.historicalData.storage_price || []).concat(stationData.price);
            this.historicalData.storage_load = (this.historicalData.storage_load || []).concat(stationData.load);
        });
        
        // 生成充电站数据
        chargingStations.forEach(stationId => {
            const stationData = {
                generation: this.generateSampleData(
                    startDate,
                    endDate,
                    stationId,
                    '充电量',
            'kWh',
            [2000, 5000],
            0.08
                ),
                price: this.generateSampleData(
                    startDate,
                    endDate,
                    stationId,
                    '充电价格',
                    '元/kWh',
                    [0.8, 2.5],
                    0.05
                ),
                load: this.generateSampleData(
                    startDate,
                    endDate,
                    stationId,
                    '充电负荷',
                    'kW',
                    [1500, 4000],
                    0.08
                )
            };
            
            this.historicalData.charging_generation = (this.historicalData.charging_generation || []).concat(stationData.generation);
            this.historicalData.charging_price = (this.historicalData.charging_price || []).concat(stationData.price);
            this.historicalData.charging_load = (this.historicalData.charging_load || []).concat(stationData.load);
        });
        
        console.log('示例数据加载完成，数据集:', Object.keys(this.historicalData));
        console.log('风电数据示例:', this.historicalData.wind_generation.slice(0, 2));
        
        // 打印每个数据集的记录数
        Object.entries(this.historicalData).forEach(([key, data]) => {
            console.log(`${key} 数据集记录数:`, data.length);
        });
    },
    
    // 生成示例数据
    generateSampleData(startDate, endDate, stationId, fieldName, unit, valueRange, anomalyRate = 0.05) {
        const start = new Date(startDate);
        const end = new Date(endDate);
        const data = [];
        
        // 循环生成每天数据
        for (let date = new Date(start); date <= end; date.setDate(date.getDate() + 1)) {
            // 基础值：在范围内的随机值
            const baseValue = valueRange[0] + Math.random() * (valueRange[1] - valueRange[0]);
            
            // 季节性波动：冬季和夏季高，春秋低
            const month = date.getMonth();
            let seasonalFactor = 1;
            if (month <= 1 || month >= 11) { // 冬季
                seasonalFactor = 1.2;
            } else if (month >= 5 && month <= 8) { // 夏季
                seasonalFactor = 1.1;
            } else { // 春秋
                seasonalFactor = 0.9;
            }
            
            // 周末效应：周末略低
            const dayOfWeek = date.getDay();
            const weekendFactor = (dayOfWeek === 0 || dayOfWeek === 6) ? 0.85 : 1;
            
            // 计算最终值
            let value = baseValue * seasonalFactor * weekendFactor;
            
            // 添加随机波动
            value = value * (0.95 + Math.random() * 0.1);
            
            // 四舍五入到两位小数
            value = Math.round(value * 100) / 100;
            
            // 确定是否为异常值
            const isAnomaly = Math.random() < anomalyRate;
            
            // 如果是异常值，则大幅改变值
            if (isAnomaly) {
                // 50%概率大幅增加，50%概率大幅减少
                if (Math.random() < 0.5) {
                    value = value * (1.5 + Math.random() * 0.5); // 增加50%-100%
                } else {
                    value = value * (0.3 + Math.random() * 0.3); // 减少40%-70%
                }
                value = Math.round(value * 100) / 100;
            }
            
            // 格式化日期
            const formattedDate = date.toISOString().split('T')[0];
            
            // 添加数据点
            data.push({
                date: formattedDate,
                value: value,
                unit: unit,
                field: fieldName,
                stationId: stationId,
                status: isAnomaly ? 'anomaly' : 'normal'
            });
        }
        
        return data;
    },

    // 导出为CSV
    exportAsCSV(data, params) {
        // 获取场景和数据类型名称
        const sceneTypeName = this.getSceneTypeLabel(params.sceneType);
        const dataTypeName = this.getDataTypeLabel(params.dataType);
        
        // 准备导出选项对话框
        const exportDialog = document.createElement('div');
        exportDialog.className = 'modal fade';
        exportDialog.id = 'exportOptionsDialog';
        exportDialog.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content bg-dark text-light">
                    <div class="modal-header">
                        <h5 class="modal-title">导出选项</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <form id="exportOptionsForm">
                            <div class="mb-3">
                                <label class="form-label">导出格式</label>
                                <select class="form-select bg-dark text-light" id="exportFormat">
                                    <option value="csv">CSV</option>
                                    <option value="excel">Excel</option>
                                    <option value="json">JSON</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <label class="form-label">时间格式</label>
                                <select class="form-select bg-dark text-light" id="dateFormat">
                                    <option value="yyyy-MM-dd">YYYY-MM-DD</option>
                                    <option value="yyyy/MM/dd">YYYY/MM/DD</option>
                                    <option value="dd/MM/yyyy">DD/MM/YYYY</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <label class="form-label">数值精度</label>
                                <select class="form-select bg-dark text-light" id="precision">
                                    <option value="0">整数</option>
                                    <option value="1">1位小数</option>
                                    <option value="2" selected>2位小数</option>
                                    <option value="3">3位小数</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="includeMetadata" checked>
                                    <label class="form-check-label">包含元数据</label>
                                </div>
                            </div>
                            <div class="mb-3">
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="includeQuality" checked>
                                    <label class="form-check-label">包含数据质量信息</label>
                                </div>
                            </div>
                        </form>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
                        <button type="button" class="btn btn-primary" id="confirmExport">确认导出</button>
                    </div>
                </div>
            </div>
        `;
        
        // 添加对话框到文档
        document.body.appendChild(exportDialog);
        
        // 初始化Bootstrap模态框
        const modal = new bootstrap.Modal(exportDialog);
        modal.show();
        
        // 绑定导出确认事件
        document.getElementById('confirmExport').addEventListener('click', () => {
            const format = document.getElementById('exportFormat').value;
            const dateFormat = document.getElementById('dateFormat').value;
            const precision = parseInt(document.getElementById('precision').value);
            const includeMetadata = document.getElementById('includeMetadata').checked;
            const includeQuality = document.getElementById('includeQuality').checked;
            
            // 根据选择的格式执行相应的导出
            switch (format) {
                case 'csv':
                    this.exportToCSV(data, params, { dateFormat, precision, includeMetadata, includeQuality });
                    break;
                case 'excel':
                    this.exportToExcel(data, params, { dateFormat, precision, includeMetadata, includeQuality });
                    break;
                case 'json':
                    this.exportToJSON(data, params, { dateFormat, precision, includeMetadata, includeQuality });
                    break;
            }
            
            // 关闭对话框
            modal.hide();
            
            // 移除对话框
            exportDialog.remove();
        });
    },
    
    // 导出为CSV格式
    exportToCSV(data, params, options) {
        // 准备元数据
        const metadata = options.includeMetadata ? [
            ['导出时间', new Date().toLocaleString()],
            ['数据类型', this.getDataTypeLabel(params.dataType)],
            ['场景类型', this.getSceneTypeLabel(params.sceneType)],
            ['时间范围', `${params.startDate} 至 ${params.endDate}`],
            ['时间粒度', this.getGranularityLabel(params.granularity)],
            ['站点', params.station ? this.getStationName(params.sceneType, params.station) : '全部'],
            ['', ''] // 空行分隔
        ] : [];
        
        // 准备表头
        const headers = ['时间', '数值', '单位', '状态'];
        if (options.includeQuality) {
            headers.push('质量标记');
        }
        
        // 生成CSV内容
        let csvContent = '\uFEFF'; // UTF-8 BOM
        
        // 添加元数据
        if (options.includeMetadata) {
            metadata.forEach(row => {
                csvContent += row.join(',') + '\n';
            });
        }
        
        // 添加表头
        csvContent += headers.join(',') + '\n';
        
        // 添加数据行
        data.forEach(row => {
            const formattedDate = this.formatDateForExport(row.date, options.dateFormat);
            const formattedValue = row.value.toFixed(options.precision);
            const values = [
                `"${formattedDate}"`,
                formattedValue,
                row.unit,
                row.status === 'normal' ? '正常' : '异常'
            ];
            
            if (options.includeQuality) {
                values.push(this.getQualityMark(row));
            }
            
            csvContent += values.join(',') + '\n';
        });
        
        // 创建Blob对象
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        
        // 生成文件名
        const fileName = `${sceneTypeName}_${dataTypeName}_${params.startDate}_${params.endDate}.csv`;
        
        // 下载文件
        this.downloadFile(blob, fileName);
    },
    
    // 导出为Excel格式
    exportToExcel(data, params, options) {
        // 使用SheetJS库导出Excel
        // 注：需要先引入SheetJS库
        if (typeof XLSX === 'undefined') {
            console.error('SheetJS库未加载');
            alert('导出Excel需要加载SheetJS库，请确保已引入相关依赖');
            return;
        }
        
        // 准备工作表数据
        const wsData = [];
        
        // 添加元数据
        if (options.includeMetadata) {
            wsData.push(['导出时间', new Date().toLocaleString()]);
            wsData.push(['数据类型', this.getDataTypeLabel(params.dataType)]);
            wsData.push(['场景类型', this.getSceneTypeLabel(params.sceneType)]);
            wsData.push(['时间范围', `${params.startDate} 至 ${params.endDate}`]);
            wsData.push(['时间粒度', this.getGranularityLabel(params.granularity)]);
            wsData.push(['站点', params.station ? this.getStationName(params.sceneType, params.station) : '全部']);
            wsData.push([]); // 空行分隔
        }
        
        // 添加表头
        const headers = ['时间', '数值', '单位', '状态'];
        if (options.includeQuality) {
            headers.push('质量标记');
        }
        wsData.push(headers);
        
        // 添加数据行
        data.forEach(row => {
            const formattedDate = this.formatDateForExport(row.date, options.dateFormat);
            const formattedValue = row.value.toFixed(options.precision);
            const rowData = [
                formattedDate,
                formattedValue,
                row.unit,
                row.status === 'normal' ? '正常' : '异常'
            ];
            
            if (options.includeQuality) {
                rowData.push(this.getQualityMark(row));
            }
            
            wsData.push(rowData);
        });
        
        // 创建工作簿
        const wb = XLSX.utils.book_new();
        const ws = XLSX.utils.aoa_to_sheet(wsData);
        
        // 设置列宽
        const colWidths = [
            { wch: 20 }, // 时间列
            { wch: 12 }, // 数值列
            { wch: 8 },  // 单位列
            { wch: 8 },  // 状态列
            { wch: 12 }  // 质量标记列
        ];
        ws['!cols'] = colWidths;
        
        // 添加工作表到工作簿
        XLSX.utils.book_append_sheet(wb, ws, '历史数据');
        
        // 生成文件名
        const fileName = `${sceneTypeName}_${dataTypeName}_${params.startDate}_${params.endDate}.xlsx`;
        
        // 导出文件
        XLSX.writeFile(wb, fileName);
    },
    
    // 导出为JSON格式
    exportToJSON(data, params, options) {
        // 准备导出数据
        const exportData = {
            data: data.map(row => ({
                date: this.formatDateForExport(row.date, options.dateFormat),
                value: Number(row.value.toFixed(options.precision)),
                unit: row.unit,
                status: row.status
            }))
        };
        
        // 添加元数据
        if (options.includeMetadata) {
            exportData.metadata = {
                exportTime: new Date().toISOString(),
                dataType: this.getDataTypeLabel(params.dataType),
                sceneType: this.getSceneTypeLabel(params.sceneType),
                timeRange: {
                    start: params.startDate,
                    end: params.endDate
                },
                granularity: params.granularity,
                station: params.station ? this.getStationName(params.sceneType, params.station) : 'all'
            };
        }
        
        // 添加质量信息
        if (options.includeQuality) {
            exportData.data = exportData.data.map(row => ({
                ...row,
                qualityMark: this.getQualityMark(row)
            }));
        }
        
        // 创建Blob对象
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json;charset=utf-8;' });
        
        // 生成文件名
        const fileName = `${sceneTypeName}_${dataTypeName}_${params.startDate}_${params.endDate}.json`;
        
        // 下载文件
        this.downloadFile(blob, fileName);
    },
    
    // 格式化导出日期
    formatDateForExport(dateStr, format) {
        const date = new Date(dateStr);
        const year = date.getFullYear();
        const month = (date.getMonth() + 1).toString().padStart(2, '0');
        const day = date.getDate().toString().padStart(2, '0');
        
        switch (format) {
            case 'yyyy-MM-dd':
                return `${year}-${month}-${day}`;
            case 'yyyy/MM/dd':
                return `${year}/${month}/${day}`;
            case 'dd/MM/yyyy':
                return `${day}/${month}/${year}`;
            default:
                return dateStr;
        }
    },
    
    // 获取数据质量标记
    getQualityMark(row) {
        if (row.status === 'normal') {
            return 'A';  // 正常数据
        } else if (row.value === null || row.value === undefined) {
            return 'M';  // 缺失值
        } else if (row.status === 'anomaly') {
            return 'E';  // 异常值
        } else {
            return 'U';  // 未知状态
        }
    },
    
    // 通用文件下载函数
    downloadFile(blob, fileName) {
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = fileName;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(link.href);
    },

    // 获取当前数据的辅助函数
    getCurrentData() {
        // 从表单中获取当前查询参数
        const form = document.getElementById('historyDataQueryForm');
        if (!form) return null;
        
        const formData = new FormData(form);
        const params = {
            sceneType: formData.get('sceneType'),
            dataType: formData.get('dataType'),
            startDate: formData.get('startDate'),
            endDate: formData.get('endDate'),
            granularity: formData.get('granularity'),
            station: formData.get('station'),
            includeQuality: formData.get('includeQualityData') === 'on'
        };
        
        // 查询数据
        return {
            data: this.queryHistoricalData(params)?.data || [],
            params: params
        };
    },
};
