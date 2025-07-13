/**
 * Header Initialization Script
 * This script initializes all header-related functionality
 */
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Feather icons
    if (typeof feather !== 'undefined') {
        feather.replace();
    }

    // Initialize bootstrap dropdowns properly
    const dropdownElementList = [].slice.call(document.querySelectorAll('[data-bs-toggle="dropdown"]'));
    dropdownElementList.map(function (dropdownToggleEl) {
        return new bootstrap.Dropdown(dropdownToggleEl);
    });

    // Initialize mode switcher
    const modeSwitchBtn = document.getElementById('mode-setting-btn');
    if (modeSwitchBtn) {
        modeSwitchBtn.addEventListener('click', function() {
            const body = document.body;
            const currentMode = body.getAttribute('data-layout-mode');
            const newMode = currentMode === 'dark' ? 'light' : 'dark';
            
            // Update layout mode
            body.setAttribute('data-layout-mode', newMode);
            
            // Update topbar and sidebar colors
            body.setAttribute('data-topbar', newMode);
            body.setAttribute('data-sidebar', newMode);
            
            // Store preference
            localStorage.setItem('layout-mode', newMode);
            
            // Update the settings in the right sidebar if it exists
            if (newMode === 'dark') {
                const darkModeRadio = document.getElementById('layout-mode-dark');
                if (darkModeRadio) darkModeRadio.checked = true;
                
                const topbarDarkRadio = document.getElementById('topbar-color-dark');
                if (topbarDarkRadio) topbarDarkRadio.checked = true;
                
                const sidebarDarkRadio = document.getElementById('sidebar-color-dark');
                if (sidebarDarkRadio) sidebarDarkRadio.checked = true;
            } else {
                const lightModeRadio = document.getElementById('layout-mode-light');
                if (lightModeRadio) lightModeRadio.checked = true;
                
                const topbarLightRadio = document.getElementById('topbar-color-light');
                if (topbarLightRadio) topbarLightRadio.checked = true;
                
                const sidebarLightRadio = document.getElementById('sidebar-color-light');
                if (sidebarLightRadio) sidebarLightRadio.checked = true;
            }
        });
    }

    // Language switcher
    const languageItems = document.querySelectorAll('.language');
    if (languageItems.length > 0) {
        languageItems.forEach(function(item) {
            item.addEventListener('click', function(e) {
                e.preventDefault();
                const lang = this.getAttribute('data-lang');
                const headerLangImg = document.getElementById('header-lang-img');
                
                // Set the language flag image
                if (headerLangImg) {
                    if (lang === 'en') {
                        headerLangImg.src = 'static/picture/cn.png';
                    } else if (lang === 'sp') {
                        headerLangImg.src = 'static/picture/spain.jpg';
                    } else if (lang === 'gr') {
                        headerLangImg.src = 'static/picture/germany.jpg';
                    } else if (lang === 'it') {
                        headerLangImg.src = 'static/picture/italy.jpg';
                    } else if (lang === 'ru') {
                        headerLangImg.src = 'static/picture/russia.jpg';
                    }
                }
                
                // Store language preference
                localStorage.setItem('language', lang);
            });
        });
    }

    // Initialize vertical menu toggle
    const verticalMenuBtn = document.getElementById('vertical-menu-btn');
    if (verticalMenuBtn) {
        verticalMenuBtn.addEventListener('click', function(e) {
            e.preventDefault();
            document.body.classList.toggle('sidebar-enable');
            
            // Handle sidebar size for different screen sizes
            if (window.innerWidth >= 992) {
                const currentSidebarSize = document.body.getAttribute('data-sidebar-size');
                document.body.setAttribute('data-sidebar-size', 
                    (currentSidebarSize === 'sm' || !currentSidebarSize) ? 'lg' : 'sm');
            }
        });
    }

    // Right sidebar toggle
    const rightBarToggles = document.querySelectorAll('.right-bar-toggle');
    if (rightBarToggles.length > 0) {
        rightBarToggles.forEach(function(toggle) {
            toggle.addEventListener('click', function(e) {
                document.body.classList.toggle('right-bar-enabled');
            });
        });
        
        // Close right sidebar when clicking outside
        document.addEventListener('click', function(e) {
            if (!e.target.closest('.right-bar-toggle') && 
                !e.target.closest('.right-bar')) {
                document.body.classList.remove('right-bar-enabled');
            }
        });
    }

    // Make sure all icons are properly rendered
    setTimeout(function() {
        if (typeof feather !== 'undefined') {
            feather.replace();
        }
    }, 300);
}); 