// 表格更新管理器
const PredictionTableManager = {
    // 更新表格数据
    updateTable: function(startTime, endTime, granularity) {
        const tbody = document.querySelector('#predictionDetailsTable tbody');
        if (!tbody) return;

        // 生成模拟数据
        const data = this.generateData(startTime, endTime, granularity);
        
        // 清空现有内容
        tbody.innerHTML = '';
        
        // 添加新数据
        data.forEach(item => {
            const row = document.createElement('tr');
            const errorRate = ((item.predicted - item.actual) / item.actual * 100).toFixed(2);
            const statusClass = Math.abs(errorRate) <= 5 ? 'success' : 
                              Math.abs(errorRate) <= 10 ? 'warning' : 'danger';
            
            row.innerHTML = `
                <td>${this.formatTime(item.time, granularity)}</td>
                <td>${item.predicted.toFixed(2)} MW</td>
                <td>${item.actual.toFixed(2)} MW</td>
                <td>${errorRate}%</td>
                <td><span class="badge bg-${statusClass}">
                    ${Math.abs(errorRate) <= 5 ? '正常' : 
                      Math.abs(errorRate) <= 10 ? '警告' : '异常'}
                </span></td>
            `;
            tbody.appendChild(row);
        });
    },

    // 生成模拟数据
    generateData: function(startTime, endTime, granularity) {
        const data = [];
        const start = new Date(startTime);
        const end = new Date(endTime);
        let interval;
        
        switch(granularity) {
            case 'hour':
                interval = 60 * 60 * 1000;
                break;
            case 'day':
                interval = 24 * 60 * 60 * 1000;
                break;
            case 'month':
                interval = 30 * 24 * 60 * 60 * 1000;
                break;
            case 'year':
                interval = 365 * 24 * 60 * 60 * 1000;
                break;
            default:
                interval = 60 * 60 * 1000;
        }

        for (let time = start; time <= end; time = new Date(time.getTime() + interval)) {
            const hour = time.getHours();
            const baseValue = 1000 + Math.sin((hour - 6) * Math.PI / 12) * 500;
            const predicted = baseValue * (1 + (Math.random() - 0.5) * 0.2);
            const actual = predicted * (1 + (Math.random() - 0.5) * 0.1);
            
            data.push({
                time: new Date(time),
                predicted: predicted,
                actual: actual
            });
        }
        
        return data;
    },

    // 格式化时间显示
    formatTime: function(date, granularity) {
        const options = {
            hour: { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' },
            day: { year: 'numeric', month: '2-digit', day: '2-digit' },
            month: { year: 'numeric', month: 'long' },
            year: { year: 'numeric' }
        };
        
        return date.toLocaleDateString('zh-CN', options[granularity] || options.hour);
    },

    // 初始化表格
    init: function() {
        // 初始化表格数据
        const now = new Date();
        const dayStart = new Date(now);
        dayStart.setHours(0, 0, 0, 0);
        const dayEnd = new Date(now);
        dayEnd.setHours(23, 59, 59, 999);
        this.updateTable(dayStart, dayEnd, 'day');

        // 绑定时间粒度切换事件
        document.querySelectorAll('.table-period-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.table-period-btn').forEach(b => 
                    b.classList.remove('active'));
                e.target.classList.add('active');
                
                let startTime, endTime;
                if (document.querySelector('#solarpowerTab').classList.contains('show')) {
                    startTime = document.getElementById('solarStartTime').value;
                    endTime = document.getElementById('solarEndTime').value;
                } else {
                    startTime = document.getElementById('windStartTime').value;
                    endTime = document.getElementById('windEndTime').value;
                }
                
                if (startTime && endTime) {
                    this.updateTable(startTime, endTime, e.target.dataset.period);
                }
            });
        });

        // 绑定查询按钮事件
        document.getElementById('solarQuery')?.addEventListener('click', () => {
            const startTime = document.getElementById('solarStartTime').value;
            const endTime = document.getElementById('solarEndTime').value;
            const granularity = document.getElementById('solarGranularity').value;
            
            if (startTime && endTime) {
                this.updateTable(startTime, endTime, granularity);
            }
        });

        document.getElementById('windQuery')?.addEventListener('click', () => {
            const startTime = document.getElementById('windStartTime').value;
            const endTime = document.getElementById('windEndTime').value;
            const granularity = document.getElementById('windGranularity').value;
            
            if (startTime && endTime) {
                this.updateTable(startTime, endTime, granularity);
            }
        });
    }
};

// 页面加载完成后初始化表格管理器
document.addEventListener('DOMContentLoaded', function() {
    PredictionTableManager.init();
}); 