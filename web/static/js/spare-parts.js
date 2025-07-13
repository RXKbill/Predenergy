// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 初始化类别选择
    initializeCategoryTabs();
    
    // 初始化搜索功能
    initializeSearch();
    
    // 初始化事件监听器
    initializeEventListeners();

    // 初始化添加备件功能
    initializeAddPart();
});

// 初始化类别选择
function initializeCategoryTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // 移除其他标签的激活状态
            tabs.forEach(t => t.classList.remove('active'));
            // 激活当前标签
            tab.classList.add('active');
            // 根据类别筛选备件列表
            filterPartsByCategory(tab.dataset.category);
        });
    });
}

// 根据类别筛选备件
function filterPartsByCategory(category) {
    const cards = document.querySelectorAll('.spare-part-card');
    if (category === 'all') {
        cards.forEach(card => card.style.display = 'flex');
    } else {
        cards.forEach(card => {
            const partCategory = card.dataset.category;
            card.style.display = partCategory === category ? 'flex' : 'none';
        });
    }
}

// 初始化搜索功能
function initializeSearch() {
    const searchInput = document.getElementById('sparePartSearch');
    if (searchInput) {
        searchInput.addEventListener('input', handleSearch);
    }
}

// 初始化事件监听器
function initializeEventListeners() {
    // 申领按钮点击事件
    document.querySelectorAll('.request-btn').forEach(button => {
        button.addEventListener('click', handleRequestClick);
    });

    // 申领表单提交事件
    const submitButton = document.getElementById('submitRequest');
    if (submitButton) {
        submitButton.addEventListener('click', handleRequestSubmit);
    }
}

// 处理申领按钮点击
function handleRequestClick(event) {
    const button = event.currentTarget;
    const partId = button.dataset.partId;
    const card = button.closest('.spare-part-card');
    const partName = card.querySelector('.part-name').textContent;
    const currentStock = parseInt(card.querySelector('.stock-value').textContent);
    
    // 设置模态框中的备件信息
    const requestModal = new bootstrap.Modal(document.getElementById('requestModal'));
    document.getElementById('requestPartName').value = partName;
    
    // 设置数量输入框的最大值
    const quantityInput = document.getElementById('requestQuantity');
    quantityInput.max = currentStock;
    quantityInput.value = '1';
    
    // 清空申领原因
    document.getElementById('requestReason').value = '';
    
    // 存储当前申领的备件ID
    document.getElementById('requestForm').dataset.partId = partId;
    
    requestModal.show();
}

// 处理申领表单提交
async function handleRequestSubmit() {
    const form = document.getElementById('requestForm');
    const partId = form.dataset.partId;
    const quantity = document.getElementById('requestQuantity').value;
    const reason = document.getElementById('requestReason').value;
    
    // 表单验证
    if (!quantity || quantity < 1) {
        showToast('请输入有效的申领数量', 'error');
        return;
    }
    
    if (!reason.trim()) {
        showToast('请输入申领原因', 'error');
        return;
    }
    
    try {
        // 这里应该调用后端API
        // 模拟API调用
        await simulateApiCall({
            partId,
            quantity: parseInt(quantity),
            reason
        });
        
        // 更新库存显示
        updateStockDisplay(partId, quantity);
        
        // 关闭模态框
        const requestModal = bootstrap.Modal.getInstance(document.getElementById('requestModal'));
        requestModal.hide();
        
        // 显示成功提示
        showToast('申领成功', 'success');
        
    } catch (error) {
        showToast(error.message || '申领失败，请重试', 'error');
    }
}

// 更新库存显示
function updateStockDisplay(partId, requestedQuantity) {
    const card = document.querySelector(`.spare-part-card[data-part-id="${partId}"]`);
    const stockElement = card.querySelector('.stock-value');
    const currentStock = parseInt(stockElement.textContent);
    const newStock = currentStock - parseInt(requestedQuantity);
    
    stockElement.textContent = newStock;
    
    // 更新库存状态样式
    const stockContainer = stockElement.closest('.part-stock');
    stockContainer.classList.remove('warning', 'danger');
    
    if (newStock <= 5) {
        stockContainer.classList.add('danger');
    } else if (newStock <= 10) {
        stockContainer.classList.add('warning');
    }
}

// 处理搜索功能
function handleSearch(event) {
    const searchTerm = event.target.value.toLowerCase();
    const cards = document.querySelectorAll('.spare-part-card');
    
    cards.forEach(card => {
        const name = card.querySelector('.part-name').textContent.toLowerCase();
        const number = card.querySelector('.part-number').textContent.toLowerCase();
        const isMatch = name.includes(searchTerm) || number.includes(searchTerm);
        
        card.style.display = isMatch ? 'flex' : 'none';
    });
}

// 显示提示信息
function showToast(message, type = 'info') {
    // 定义不同类型的样式
    const styles = {
        success: {
            background: '#198754',  // 绿色
            color: '#fff'
        },
        error: {
            background: '#dc3545',  // 红色
            color: '#fff'
        },
        info: {
            background: '#0dcaf0',  // 蓝色
            color: '#fff'
        },
        warning: {
            background: '#ffc107',  // 黄色
            color: '#000'
        }
    };

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.style.position = 'fixed';
    toast.style.bottom = '100px';
    toast.style.right = '20px';
    toast.style.zIndex = '1050';
    toast.style.minWidth = '200px';
    toast.style.background = styles[type].background;
    toast.style.color = styles[type].color;
    toast.style.borderRadius = '4px';
    toast.style.boxShadow = '0 0.5rem 1rem rgba(0, 0, 0, 0.15)';
    
    toast.innerHTML = `
        <div class="d-flex align-items-center justify-content-between p-3">
            <div class="toast-body" style="padding: 0;">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white ms-2" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    document.body.appendChild(toast);
    const bsToast = new bootstrap.Toast(toast, {
        delay: 3000,
        animation: true
    });
    bsToast.show();
    
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

// 模拟API调用
function simulateApiCall(data) {
    return new Promise((resolve, reject) => {
        // 模拟网络延迟
        setTimeout(() => {
            // 模拟成功率80%
            if (Math.random() > 0.2) {
                resolve({
                    success: true,
                    message: '申领成功'
                });
            } else {
                reject(new Error('网络错误，请重试'));
            }
        }, 1000);
    });
}

// 初始化添加备件功能
function initializeAddPart() {
    const submitNewPartBtn = document.getElementById('submitNewPart');
    if (submitNewPartBtn) {
        submitNewPartBtn.addEventListener('click', handleAddPart);
    }

    // 初始化表单验证
    initializeFormValidation();
}

// 初始化表单验证
function initializeFormValidation() {
    const stockInput = document.getElementById('newPartStock');
    if (stockInput) {
        stockInput.addEventListener('input', function() {
            if (this.value < 0) {
                this.value = 0;
            }
        });
    }
}

// 处理添加备件
async function handleAddPart() {
    // 获取表单数据
    const newPart = {
        name: document.getElementById('newPartName').value,
        category: document.getElementById('newPartCategory').value,
        spec: document.getElementById('newPartSpec').value,
        stock: document.getElementById('newPartStock').value,
        unit: document.getElementById('newPartUnit').value,
        location: document.getElementById('newPartLocation').value
    };

    // 表单验证
    if (!validateNewPart(newPart)) {
        return;
    }

    try {
        // 显示加载状态
        const submitBtn = document.getElementById('submitNewPart');
        const originalText = submitBtn.innerHTML;
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>添加中...';

        // 模拟API调用
        await simulateAddPartApi(newPart);

        // 添加新备件卡片到列表
        addPartToList(newPart);

        // 关闭模态框
        const addModal = bootstrap.Modal.getInstance(document.getElementById('addPartModal'));
        addModal.hide();

        // 清空表单
        document.getElementById('addPartForm').reset();

        // 显示成功提示
        showToast('备件添加成功', 'success');

    } catch (error) {
        showToast(error.message || '添加失败，请重试', 'error');
    } finally {
        // 恢复按钮状态
        const submitBtn = document.getElementById('submitNewPart');
        submitBtn.disabled = false;
        submitBtn.innerHTML = '确认添加';
    }
}

// 验证新备件数据
function validateNewPart(part) {
    if (!part.name.trim()) {
        showToast('请输入备件名称', 'error');
        document.getElementById('newPartName').focus();
        return false;
    }
    if (!part.spec.trim()) {
        showToast('请输入规格型号', 'error');
        document.getElementById('newPartSpec').focus();
        return false;
    }
    if (!part.stock || isNaN(part.stock) || parseInt(part.stock) < 0) {
        showToast('请输入有效的初始库存', 'error');
        document.getElementById('newPartStock').focus();
        return false;
    }
    if (!part.unit.trim()) {
        showToast('请输入库存单位', 'error');
        document.getElementById('newPartUnit').focus();
        return false;
    }
    if (!part.location.trim()) {
        showToast('请输入存放位置', 'error');
        document.getElementById('newPartLocation').focus();
        return false;
    }
    return true;
}

// 添加备件到列表
function addPartToList(part) {
    const partsList = document.getElementById('sparePartsList');
    const newPartId = generatePartId();

    const partCard = document.createElement('div');
    partCard.className = 'spare-part-card';
    partCard.dataset.category = part.category;
    partCard.dataset.partId = newPartId;

    partCard.innerHTML = `
        <div class="part-info">
            <div class="part-name">${part.name}</div>
            <div class="part-number">${newPartId}</div>
            <div class="part-stock">
                <span class="stock-label">库存:</span>
                <span class="stock-value">${part.stock}</span>
                <span class="stock-unit">${part.unit}</span>
            </div>
            <div class="part-details">
                <span class="detail-item">规格: ${part.spec}</span>
                <span class="detail-item">位置: ${part.location}</span>
            </div>
        </div>
        <div class="part-actions">
            <button class="btn btn-sm btn-primary request-btn" data-part-id="${newPartId}">申领</button>
            <button class="btn btn-sm btn-outline-secondary details-btn" data-part-id="${newPartId}">
                <i class="mdi mdi-information"></i>
            </button>
        </div>
    `;

    // 为新添加的申领按钮绑定事件
    const requestBtn = partCard.querySelector('.request-btn');
    requestBtn.addEventListener('click', handleRequestClick);

    // 添加到列表开头
    if (partsList.firstChild) {
        partsList.insertBefore(partCard, partsList.firstChild);
    } else {
        partsList.appendChild(partCard);
    }
}

// 生成备件ID
function generatePartId() {
    const prefix = 'SPR';
    const randomNum = Math.floor(Math.random() * 10000).toString().padStart(4, '0');
    return `${prefix}-${randomNum}`;
}

// 模拟添加备件API调用
function simulateAddPartApi(data) {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            if (Math.random() > 0.1) { // 90%成功率
                resolve({
                    success: true,
                    message: '备件添加成功'
                });
            } else {
                reject(new Error('网络错误，请重试'));
            }
        }, 800);
    });
} 