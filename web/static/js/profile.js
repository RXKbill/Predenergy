// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeProfile();
});

// 初始化个人页面
function initializeProfile() {
    // 绑定头像编辑事件
    const avatarEdit = document.querySelector('.avatar-edit');
    if (avatarEdit) {
        avatarEdit.addEventListener('click', handleAvatarEdit);
    }

    // 绑定设置按钮点击事件
    const settingsBtn = document.getElementById('settingsBtn');
    if (settingsBtn) {
        settingsBtn.addEventListener('click', function() {
            handleSettings();
        });
    }

    // 绑定退出登录按钮事件
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }

    // 绑定菜单项点击事件
    document.querySelectorAll('.menu-item').forEach(item => {
        item.addEventListener('click', function() {
            const itemName = this.querySelector('.item-left span').textContent;
            handleMenuItemClick(itemName);
        });
    });
}

// 处理头像编辑
function handleAvatarEdit() {
    // 创建文件输入元素
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.style.display = 'none';
    
    // 监听文件选择
    input.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            if (file.size > 5 * 1024 * 1024) { // 5MB限制
                showNotification('图片大小不能超过5MB', 'error');
                return;
            }
            
            // 预览并上传头像
            const reader = new FileReader();
            reader.onload = function(e) {
                document.querySelector('.profile-avatar img').src = e.target.result;
                // TODO: 调用上传API
                showNotification('头像更新成功', 'success');
            };
            reader.readAsDataURL(file);
        }
    });
    
    document.body.appendChild(input);
    input.click();
    document.body.removeChild(input);
}

// 处理设置
function handleSettings() {
    showModal('系统设置', `
        <div class="settings-form">
            <div class="form-group mb-3">
                <label class="form-label">深色模式</label>
                <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" id="darkModeSwitch">
                    <label class="form-check-label" for="darkModeSwitch">启用</label>
                </div>
            </div>
            <div class="form-group mb-3">
                <label class="form-label">消息通知</label>
                <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" id="notificationSwitch" checked>
                    <label class="form-check-label" for="notificationSwitch">启用</label>
                </div>
            </div>
            <div class="form-group mb-3">
                <label class="form-label">字体大小</label>
                <select class="form-select" id="fontSize">
                    <option value="small">小</option>
                    <option value="medium" selected>中</option>
                    <option value="large">大</option>
                </select>
            </div>
            <div class="form-group">
                <label class="form-label">缓存大小</label>
                <div class="d-flex align-items-center">
                    <span class="text-muted me-2">23.5MB</span>
                    <button class="btn btn-sm btn-outline-danger" onclick="clearCache()">清除缓存</button>
                </div>
            </div>
        </div>
    `);
}

// 处理退出登录
function handleLogout() {
    showModal('退出登录', `
        <p class="mb-0">确定要退出登录吗？</p>
    `, [
        {
            text: '取消',
            class: 'btn-secondary',
            handler: (modal) => {
                modal.hide();
            }
        },
        {
            text: '确定退出',
            class: 'btn-danger',
            handler: (modal) => {
                // TODO: 调用退出登录API
                showNotification('正在退出登录...', 'info');
                setTimeout(() => {
                    window.location.href = 'app-mobile_login.html';
                }, 1000);
            }
        }
    ]);
}

// 处理菜单项点击
function handleMenuItemClick(itemName) {
    switch(itemName) {
        case '实名认证':
            showModal('实名认证', `
                <div class="text-center">
                    <i class="mdi mdi-check-circle text-success" style="font-size: 48px;"></i>
                    <h5 class="mt-2">已完成实名认证</h5>
                    <p class="text-muted mb-0">认证时间：2024-01-15</p>
                </div>
            `);
            break;
            
        case '我的证件':
            showModal('证件信息', `
                <div class="certificate-info">
                    <div class="mb-3">
                        <h6>高级维修师证书</h6>
                        <p class="text-muted mb-1">证书编号：5002********6873</p>
                        <p class="text-muted mb-1">发证机构：国家能源局</p>
                        <p class="text-muted mb-0">有效期至：2026-12-31</p>
                    </div>
                    <div class="text-center">
                        <img src="static/picture/certificate.jpg" alt="证书图片" class="img-fluid rounded">
                    </div>
                </div>
            `);
            break;
            
        case '手机号码':
            showModal('修改手机号码', `
                <form id="phoneForm">
                    <div class="mb-3">
                        <label class="form-label">当前手机号</label>
                        <input type="text" class="form-control" value="182****95333" disabled>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">新手机号</label>
                        <input type="tel" class="form-control" required pattern="[0-9]{11}">
                    </div>
                    <div class="mb-3">
                        <label class="form-label">验证码</label>
                        <div class="input-group">
                            <input type="text" class="form-control" required>
                            <button type="button" class="btn btn-outline-primary" onclick="sendVerificationCode(this)">
                                获取验证码
                            </button>
                        </div>
                    </div>
                </form>
            `, [
                {
                    text: '取消',
                    class: 'btn-secondary',
                    handler: (modal) => modal.hide()
                },
                {
                    text: '确认修改',
                    class: 'btn-primary',
                    handler: (modal) => {
                        const form = document.getElementById('phoneForm');
                        if (form.checkValidity()) {
                            showNotification('手机号码修改成功', 'success');
                            modal.hide();
                        } else {
                            form.reportValidity();
                        }
                    }
                }
            ]);
            break;
            
        case '维修历史':
            window.location.href = 'maintenance-history.html';
            break;
            
        case '工作评价':
            showModal('工作评价', `
                <div class="rating-info">
                    <div class="text-center mb-4">
                        <h1 class="display-4 text-primary">4.9</h1>
                        <div class="text-warning mb-2">
                            <i class="mdi mdi-star"></i>
                            <i class="mdi mdi-star"></i>
                            <i class="mdi mdi-star"></i>
                            <i class="mdi mdi-star"></i>
                            <i class="mdi mdi-star-half"></i>
                        </div>
                        <p class="text-muted">来自142份评价</p>
                    </div>
                    <div class="rating-stats">
                        <div class="rating-item">
                            <span>专业技能</span>
                            <div class="progress">
                                <div class="progress-bar bg-success" style="width: 98%"></div>
                            </div>
                            <span>4.9</span>
                        </div>
                        <div class="rating-item">
                            <span>服务态度</span>
                            <div class="progress">
                                <div class="progress-bar bg-success" style="width: 96%"></div>
                            </div>
                            <span>4.8</span>
                        </div>
                        <div class="rating-item">
                            <span>响应速度</span>
                            <div class="progress">
                                <div class="progress-bar bg-success" style="width: 94%"></div>
                            </div>
                            <span>4.7</span>
                        </div>
                    </div>
                </div>
            `);
            break;
            
        case '专业技能':
            showModal('专业技能', `
                <div class="skills-list">
                    <div class="skill-item">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <h6 class="mb-0">光伏系统维护</h6>
                            <span class="badge bg-success">高级</span>
                        </div>
                        <div class="progress">
                            <div class="progress-bar bg-success" style="width: 95%"></div>
                        </div>
                    </div>
                    <div class="skill-item mt-3">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <h6 class="mb-0">风机系统维护</h6>
                            <span class="badge bg-success">高级</span>
                        </div>
                        <div class="progress">
                            <div class="progress-bar bg-success" style="width: 90%"></div>
                        </div>
                    </div>
                    <div class="skill-item mt-3">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <h6 class="mb-0">储能系统维护</h6>
                            <span class="badge bg-primary">中级</span>
                        </div>
                        <div class="progress">
                            <div class="progress-bar bg-primary" style="width: 85%"></div>
                        </div>
                    </div>
                </div>
            `);
            break;
            
        case '客服服务':
            showModal('联系客服', `
                <div class="text-center">
                    <i class="mdi mdi-headphones" style="font-size: 48px; color: #2196f3;"></i>
                    <h5 class="mt-3">24小时在线客服</h5>
                    <p class="text-muted">工作时间：09:00 - 21:00</p>
                    <div class="d-grid gap-2">
                        <button class="btn btn-primary" onclick="startChat()">
                            <i class="mdi mdi-message-text me-1"></i>在线咨询
                        </button>
                        <button class="btn btn-outline-primary" onclick="makeCall()">
                            <i class="mdi mdi-phone me-1"></i>电话咨询
                        </button>
                    </div>
                </div>
            `);
            break;
            
        default:
            break;
    }
}

// 显示模态框
function showModal(title, content, buttons = [{text: '关闭', class: 'btn-primary', handler: modal => modal.hide()}]) {
    const modalEl = document.createElement('div');
    modalEl.className = 'modal fade';
    modalEl.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">${title}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
                <div class="modal-footer">
                    ${buttons.map(btn => `
                        <button type="button" class="btn ${btn.class}">${btn.text}</button>
                    `).join('')}
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modalEl);
    const modal = new bootstrap.Modal(modalEl);
    
    // 绑定按钮事件
    const footerBtns = modalEl.querySelectorAll('.modal-footer .btn');
    footerBtns.forEach((btn, index) => {
        btn.addEventListener('click', () => buttons[index].handler(modal));
    });
    
    // 监听模态框关闭事件
    modalEl.addEventListener('hidden.bs.modal', () => {
        document.body.removeChild(modalEl);
    });
    
    modal.show();
    return modal;
}

// 发送验证码
function sendVerificationCode(btn) {
    let countdown = 60;
    btn.disabled = true;
    btn.innerHTML = `${countdown}秒后重试`;
    
    const timer = setInterval(() => {
        countdown--;
        if (countdown > 0) {
            btn.innerHTML = `${countdown}秒后重试`;
        } else {
            clearInterval(timer);
            btn.disabled = false;
            btn.innerHTML = '获取验证码';
        }
    }, 1000);
    
    // TODO: 调用发送验证码API
    showNotification('验证码已发送', 'success');
}

// 清除缓存
function clearCache() {
    showNotification('正在清除缓存...', 'info');
    setTimeout(() => {
        showNotification('缓存已清除', 'success');
    }, 1000);
}

// 开始在线聊天
function startChat() {
    showNotification('正在连接客服系统...', 'info');
    // TODO: 跳转到聊天页面
}

// 拨打客服电话
function makeCall() {
    window.location.href = 'tel:400-123-4567';
}

// 显示通知
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show notification-toast`;
    notification.role = 'alert';
    
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    document.body.appendChild(notification);
    setTimeout(() => notification.classList.add('show'), 100);
    
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
} 