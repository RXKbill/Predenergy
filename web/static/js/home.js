// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 初始化通知系统
    const notificationSystem = new NotificationSystem();
    
    // 侧边菜单相关
    const menuToggle = document.getElementById('menuToggle');
    const sideMenu = document.createElement('div');
    sideMenu.className = 'side-menu';
    const overlay = document.createElement('div');
    overlay.className = 'overlay';
    document.body.appendChild(sideMenu);
    document.body.appendChild(overlay);

    // 初始化侧边菜单
    initializeSideMenu();

    // 搜索框功能
    const searchInput = document.querySelector('.search-input');
    searchInput.addEventListener('input', handleSearch);
    searchInput.addEventListener('focus', () => {
        // 搜索框获得焦点时的处理
        searchInput.placeholder = '请输入关键词';
    });
    searchInput.addEventListener('blur', () => {
        // 搜索框失去焦点时的处理
        searchInput.placeholder = '搜索设备、工单、资讯...';
    });

    // 扫码功能
    const qrcodeBtn = document.querySelector('.mdi-qrcode-scan');
    qrcodeBtn.addEventListener('click', handleQRCodeScan);

    // 刷新按钮功能
    const refreshBtn = document.getElementById('refreshBtn');
    refreshBtn.addEventListener('click', refreshData);

    // 通知按钮功能
    const notificationBtn = document.getElementById('notificationBtn');
    notificationBtn.addEventListener('click', showNotifications);

    // 快捷功能区域点击事件
    initializeQuickActions();

    // 功能导航区域点击事件
    initializeFeatureGrid();
});

// 通知系统类
class NotificationSystem {
    constructor() {
        this.container = document.createElement('div');
        this.container.className = 'notification-container';
        document.body.appendChild(this.container);
    }

    show(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <div class="notification-message">${message}</div>
            </div>
            <div class="notification-progress">
                <div class="notification-progress-bar"></div>
            </div>
        `;

        this.container.appendChild(notification);
        notification.classList.add('show');

        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// 初始化侧边菜单
function initializeSideMenu() {
    const sideMenu = document.querySelector('.side-menu');
    const overlay = document.querySelector('.overlay');
    const menuToggle = document.getElementById('menuToggle');

    menuToggle.addEventListener('click', () => {
        sideMenu.classList.toggle('active');
        overlay.classList.toggle('active');
    });

    overlay.addEventListener('click', () => {
        sideMenu.classList.remove('active');
        overlay.classList.remove('active');
    });
}

// 搜索功能
function handleSearch(event) {
    const searchTerm = event.target.value.toLowerCase();
    // TODO: 实现搜索逻辑
    console.log('Searching for:', searchTerm);
}

// 扫码功能
function handleQRCodeScan() {
    // 检查是否支持调用摄像头
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        // TODO: 实现扫码逻辑
        notificationSystem.show('正在打开摄像头...', 'info');
    } else {
        notificationSystem.show('设备不支持扫码功能', 'error');
    }
}

// 刷新数据
function refreshData() {
    const refreshBtn = document.getElementById('refreshBtn');
    refreshBtn.classList.add('rotating');
    
    // 模拟数据刷新
    setTimeout(() => {
        refreshBtn.classList.remove('rotating');
        notificationSystem.show('数据已更新', 'success');
    }, 1000);
}

// 显示通知列表
function showNotifications() {
    // TODO: 实现通知列表显示逻辑
    notificationSystem.show('您有2条未读消息', 'info');
}

// 初始化快捷功能区域
function initializeQuickActions() {
    const quickActions = document.querySelectorAll('.quick-action-item, .quick-action-subitem');
    quickActions.forEach(action => {
        action.addEventListener('click', (e) => {
            const actionName = e.currentTarget.querySelector('span').textContent;
            handleQuickAction(actionName);
        });
    });
}

// 处理快捷功能点击
function handleQuickAction(actionName) {
    switch(actionName) {
        case '备件管理':
            notificationSystem.show('正在打开备件管理...', 'info');
            // TODO: 跳转到备件管理页面
            window.location.href = 'app-spare-parts.html';
            break;
        case '运行报表':
            notificationSystem.show('正在生成运行报表...', 'info');
            // TODO: 跳转到运行报表页面
            window.location.href = 'app-report.html';
            break;
        case '故障预警':
            notificationSystem.show('正在查看故障预警...', 'warning');
            // TODO: 跳转到故障预警页面
            window.location.href = 'app-alert.html';
            break;
    }
}

// 初始化功能导航区域
function initializeFeatureGrid() {
    const featureItems = document.querySelectorAll('.feature-item');
    featureItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const featureName = e.currentTarget.querySelector('span').textContent;
            handleFeatureClick(featureName);
        });
    });
}

// 处理功能导航点击
function handleFeatureClick(featureName) {
    switch(featureName) {
        case '能源监测':
            window.location.href = 'app-monitor.html';
            break;
        case '预测概览':
            window.location.href = 'app-mobile_index.html';
            break;
        case '告警中心':
            window.location.href = 'app-alert.html';
            break;
        case '工单管理':
            window.location.href = 'app-workorder.html';
            break;
        case '设备定位':
            window.location.href = 'app-location.html';
            break;
        case '系统设置':
            window.location.href = 'app-settings.html';
            break;
        case '帮助中心':
            window.location.href = 'app-help.html';
            break;
        case '个人中心':
            window.location.href = 'app-profile.html';
            break;
    }
}

// 添加旋转动画类
document.head.insertAdjacentHTML('beforeend', `
    <style>
        @keyframes rotating {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        .rotating {
            animation: rotating 1s linear infinite;
        }
    </style>
`); 