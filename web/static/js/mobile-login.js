document.addEventListener('DOMContentLoaded', function() {
    initializeLogin();
});

function initializeLogin() {
    // 初始化密码显示切换
    initializePasswordToggle();
    
    // 初始化表单验证
    initializeFormValidation();
    
    // 初始化协议勾选
    initializeAgreementCheck();

    // 初始化社交登录按钮
    initializeSocialLogin();
}

// 初始化密码显示切换
function initializePasswordToggle() {
    const toggleBtn = document.getElementById('togglePassword');
    const passwordInput = document.getElementById('password');
    
    if (toggleBtn && passwordInput) {
        toggleBtn.addEventListener('click', function() {
            const type = passwordInput.getAttribute('type');
            passwordInput.setAttribute('type', type === 'password' ? 'text' : 'password');
            
            // 切换图标
            const icon = this.querySelector('i');
            icon.classList.toggle('mdi-eye-outline');
            icon.classList.toggle('mdi-eye-off-outline');
        });
    }
}

// 初始化表单验证
function initializeFormValidation() {
    const loginForm = document.getElementById('loginForm');
    const loginBtn = loginForm.querySelector('.btn-login');
    const accountInput = document.getElementById('account');
    const passwordInput = document.getElementById('password');
    const agreeCheckbox = document.getElementById('agreeTerms');
    
    // 实时验证输入
    [accountInput, passwordInput].forEach(input => {
        input.addEventListener('input', validateForm);
    });
    
    agreeCheckbox.addEventListener('change', validateForm);
    
    // 表单提交
    loginForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!validateForm()) {
            return;
        }
        
        // 显示加载状态
        loginBtn.disabled = true;
        loginBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>登录中...';
        
        // 模拟登录请求
        setTimeout(() => {
            const account = accountInput.value;
            const password = passwordInput.value;
            
            // TODO: 这里应该调用实际的登录API
            // 模拟登录成功
            handleLoginSuccess();
        }, 1500);
    });
}

// 处理登录成功
function handleLoginSuccess() {
    // 可以在这里保存登录状态、token等
    localStorage.setItem('isLoggedIn', 'true');
    
    // 显示成功提示
    showNotification('登录成功，正在跳转...', 'success');
    
    // 延迟跳转到首页
    setTimeout(() => {
        window.location.href = 'app-mobile_home.html';
    }, 1000);
}

// 表单验证
function validateForm() {
    const account = document.getElementById('account').value;
    const password = document.getElementById('password').value;
    const agreeTerms = document.getElementById('agreeTerms').checked;
    const loginBtn = document.querySelector('.btn-login');
    
    // 验证账号
    if (!account) {
        showNotification('请输入账号', 'error');
        loginBtn.disabled = true;
        return false;
    }
    
    // 验证密码
    if (!password) {
        showNotification('请输入密码', 'error');
        loginBtn.disabled = true;
        return false;
    }
    
    // 验证协议勾选
    if (!agreeTerms) {
        showNotification('请阅读并同意用户协议', 'error');
        loginBtn.disabled = true;
        return false;
    }
    
    // 验证通过
    loginBtn.disabled = false;
    return true;
}

// 初始化协议勾选
function initializeAgreementCheck() {
    const agreeCheckbox = document.getElementById('agreeTerms');
    const loginBtn = document.querySelector('.btn-login');
    
    // 默认禁用登录按钮
    loginBtn.disabled = true;
    
    agreeCheckbox.addEventListener('change', function() {
        validateForm();
    });
}

// 初始化社交登录按钮
function initializeSocialLogin() {
    const socialBtns = document.querySelectorAll('.social-btn');
    
    socialBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            // TODO: 实现社交登录逻辑
            showNotification('暂未开放，敬请期待', 'info');
        });
    });
}

// 显示通知提示
function showNotification(message, type = 'info') {
    // 创建提示元素
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white border-0 ${getTypeClass(type)}`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    // 添加到页面
    const container = document.createElement('div');
    container.className = 'toast-container position-fixed top-0 end-0 p-3';
    container.appendChild(toast);
    document.body.appendChild(container);
    
    // 显示提示
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    // 自动移除
    toast.addEventListener('hidden.bs.toast', () => {
        document.body.removeChild(container);
    });
}

// 获取通知类型对应的样式类
function getTypeClass(type) {
    const classes = {
        success: 'bg-success',
        error: 'bg-danger',
        warning: 'bg-warning',
        info: 'bg-info'
    };
    return classes[type] || classes.info;
} 