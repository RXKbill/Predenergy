// Toast通知功能
function showToast(message, type = 'success') {
    const toast = document.getElementById('recordCancelToast');
    const toastBody = document.getElementById('recordCancelToastBody');
    
    // 设置消息内容
    toastBody.textContent = message;
    
    // 设置toast样式
    toast.classList.remove('bg-success', 'bg-danger', 'bg-warning');
    toast.classList.add(`bg-${type}`);
    
    // 显示toast
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
}

// 取消记录
function cancelRecord(recordId) {
    if (confirm('确定要取消该业务记录吗？')) {
        // 这里应该调用后端API
        // 模拟API调用延迟
        setTimeout(() => {
            showToast('业务记录已成功取消', 'success');
            // 刷新表格数据
            setTimeout(() => {
                location.reload();
            }, 1500);
        }, 500);
    }
}

// 筛选功能
function applyFilter() {
    const type = document.getElementById('filter-type').value;
    const status = document.getElementById('filter-status').value;
    const startTime = document.getElementById('filter-start-time').value;
    const endTime = document.getElementById('filter-end-time').value;
    const keyword = document.getElementById('filter-keyword').value.toLowerCase();

    const rows = document.querySelectorAll('#business-records-table tbody tr');
    let visibleCount = 0;
    
    rows.forEach(row => {
        const cells = row.querySelectorAll('td');
        const rowType = cells[1].textContent;
        const rowStatus = cells[6].querySelector('.badge').textContent;
        const rowTime = cells[4].textContent;
        const searchText = [
            cells[0].textContent,
            cells[2].textContent,
            cells[3].textContent
        ].join(' ').toLowerCase();

        let show = true;

        if (type && rowType !== type) show = false;
        if (status && rowStatus !== status) show = false;
        if (startTime && new Date(rowTime) < new Date(startTime)) show = false;
        if (endTime && new Date(rowTime) > new Date(endTime)) show = false;
        if (keyword && !searchText.includes(keyword)) show = false;

        row.style.display = show ? '' : 'none';
        if (show) visibleCount++;
    });

    // 关闭筛选模态框
    const filterModal = bootstrap.Modal.getInstance(document.getElementById('filterModal'));
    filterModal.hide();

    // 显示筛选结果提示
    showToast(`筛选完成，共找到 ${visibleCount} 条记录`, visibleCount > 0 ? 'success' : 'warning');
}

// 导出记录
function exportRecords() {
    const records = Array.from(document.querySelectorAll('#business-records-table tbody tr')).map(row => {
        const cells = row.querySelectorAll('td');
        return {
            recordId: cells[0].textContent,
            businessType: cells[1].textContent,
            content: cells[2].textContent,
            target: cells[3].textContent,
            executeTime: cells[4].textContent,
            completeTime: cells[5].textContent,
            status: cells[6].querySelector('.badge').textContent
        };
    });

    try {
        const wb = XLSX.utils.book_new();
        const ws = XLSX.utils.json_to_sheet(records);
        
        const colWidths = [
            { wch: 20 }, { wch: 15 }, { wch: 30 },
            { wch: 25 }, { wch: 20 }, { wch: 20 }, { wch: 15 }
        ];
        ws['!cols'] = colWidths;

        XLSX.utils.book_append_sheet(wb, ws, "业务执行记录");

        const fileName = `业务执行记录_${new Date().toISOString().split('T')[0]}.xlsx`;
        XLSX.writeFile(wb, fileName);

        showToast('记录导出成功', 'success');
    } catch (error) {
        console.error('导出失败:', error);
        showToast('记录导出失败，请重试', 'danger');
    }
}

// 导出记录详情
function exportRecordDetail() {
    try {
        const detail = {
            recordId: document.getElementById('detail-id').textContent,
            businessType: document.getElementById('detail-type').textContent,
            content: document.getElementById('detail-content').textContent,
            target: document.getElementById('detail-target').textContent,
            executeTime: document.getElementById('detail-execute-time').textContent,
            completeTime: document.getElementById('detail-complete-time').textContent,
            status: document.getElementById('detail-status').textContent,
            result: document.getElementById('detail-result').textContent,
            timeline: Array.from(document.querySelectorAll('#detail-timeline tr')).map(row => ({
                time: row.cells[0].textContent,
                event: row.cells[1].textContent
            }))
        };

        const wb = XLSX.utils.book_new();
        
        const basicInfo = [{
            "字段": "记录ID",
            "内容": detail.recordId
        }, {
            "字段": "业务类型",
            "内容": detail.businessType
        }, {
            "字段": "执行内容",
            "内容": detail.content
        }, {
            "字段": "目标对象",
            "内容": detail.target
        }, {
            "字段": "执行时间",
            "内容": detail.executeTime
        }, {
            "字段": "完成时间",
            "内容": detail.completeTime
        }, {
            "字段": "状态",
            "内容": detail.status
        }, {
            "字段": "执行结果",
            "内容": detail.result
        }];

        const wsBasic = XLSX.utils.json_to_sheet(basicInfo);
        XLSX.utils.book_append_sheet(wb, wsBasic, "基本信息");

        const wsTimeline = XLSX.utils.json_to_sheet(detail.timeline);
        XLSX.utils.book_append_sheet(wb, wsTimeline, "执行历史");

        const fileName = `业务记录详情_${detail.recordId}.xlsx`;
        XLSX.writeFile(wb, fileName);

        showToast('记录详情导出成功', 'success');
    } catch (error) {
        console.error('导出详情失败:', error);
        showToast('记录详情导出失败，请重试', 'danger');
    }
}

// 查看记录详情
function showRecordDetail(recordId) {
    // 这里应该是从后端获取详细数据
    // 现在使用模拟数据展示
    const record = {
        id: recordId,
        type: "调度指令",
        content: "功率调整 (80kW → 120kW)",
        target: "成都高新区A3站快充桩#5",
        executeTime: "2024-03-26 09:15:23",
        completeTime: "2024-03-26 09:16:45",
        status: "已完成",
        result: "功率调整成功，目标功率120kW已达到，充电桩运行正常，响应时间1.5秒，执行效率98.5%",
        timeline: [
            { time: "2024-03-26 09:15:23", event: "开始执行业务" },
            { time: "2024-03-26 09:15:45", event: "系统验证通过，参数校验完成" },
            { time: "2024-03-26 09:16:30", event: "执行指令下发，充电桩开始功率调整" },
            { time: "2024-03-26 09:16:45", event: "业务执行完成，功率调整目标达成" }
        ]
    };

    // 填充详情模态框
    document.getElementById('detail-id').textContent = record.id;
    document.getElementById('detail-type').textContent = record.type;
    document.getElementById('detail-content').textContent = record.content;
    document.getElementById('detail-target').textContent = record.target;
    document.getElementById('detail-execute-time').textContent = record.executeTime;
    document.getElementById('detail-complete-time').textContent = record.completeTime;
    document.getElementById('detail-status').textContent = record.status;
    document.getElementById('detail-result').textContent = record.result;

    // 填充时间线
    const timelineHtml = record.timeline.map(item => `
        <tr>
            <td>${item.time}</td>
            <td${item.event.includes('完成') ? ' class="text-success"' : ''}>${item.event}</td>
        </tr>
    `).join('');
    document.getElementById('detail-timeline').innerHTML = timelineHtml;

    // 显示模态框
    const detailModal = new bootstrap.Modal(document.getElementById('recordDetailModal'));
    detailModal.show();
}

// 初始化事件监听
document.addEventListener('DOMContentLoaded', function() {
    // 绑定导出按钮事件
    document.getElementById('export-records')?.addEventListener('click', exportRecords);
    document.getElementById('export-detail')?.addEventListener('click', exportRecordDetail);

    // 绑定筛选按钮事件
    document.getElementById('apply-filter')?.addEventListener('click', applyFilter);

    // 重置筛选条件
    document.getElementById('filter-form')?.addEventListener('reset', () => {
        setTimeout(() => {
            applyFilter();
        }, 0);
    });

    // 绑定记录行点击事件
    document.querySelectorAll('#business-records-table tbody tr').forEach(row => {
        const recordId = row.querySelector('td:first-child').textContent;
        row.querySelector('.btn-primary').addEventListener('click', () => showRecordDetail(recordId));
    });

    // 绑定取消按钮事件
    document.querySelectorAll('#business-records-table tbody tr').forEach(row => {
        const cancelBtn = row.querySelector('.btn-danger');
        if (cancelBtn) {
            const recordId = row.querySelector('td:first-child').textContent;
            cancelBtn.addEventListener('click', () => cancelRecord(recordId));
        }
    });
});

// 加载总览页面内容
function loadOverviewContent() {
    $.get('templates/components/overview.html', function(data) {
        $('#overview').html(data);
        
        // 初始化业务趋势图表
        var businessTrendOptions = {
            series: [{
                name: '业务量',
                data: [30, 40, 35, 50, 49, 60, 70]
            }],
            chart: {
                type: 'line',
                height: 350
            },
            xaxis: {
                categories: ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            },
            title: {
                text: '本周业务处理趋势'
            }
        };
        new ApexCharts(document.querySelector("#business-trend-chart"), businessTrendOptions).render();

        // 初始化任务分布图表
        var taskDistributionOptions = {
            series: [44, 55, 13, 33],
            chart: {
                type: 'donut',
                height: 350
            },
            labels: ['电力调度', '能源交易', '设备维护', '充电管理'],
            title: {
                text: '任务类型分布'
            }
        };
        new ApexCharts(document.querySelector("#task-distribution-chart"), taskDistributionOptions).render();
    });
}

// 当文档加载完成时执行
$(document).ready(function() {
    // ... existing code ...
    
    // 加载总览页面内容
    loadOverviewContent();
    
    // 为导航标签添加点击事件
    $('.nav-link').on('click', function (e) {
        e.preventDefault();
        $(this).tab('show');
        
        // 根据选中的标签加载相应的内容
        var targetId = $(this).attr('href').substring(1);
        switch(targetId) {
            case 'overview':
                loadOverviewContent();
                break;
            // 其他标签的处理将在后续添加
        }
    });
}); 