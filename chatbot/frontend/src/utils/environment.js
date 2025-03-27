// frontend/src/utils/environment.js

/**
 * Checks if the code is running in an Electron environment
 * This is important for determining proper API URLs and other behaviors
 * 
 * @returns {boolean} True if running in Electron
 */
export const isElectron = () => {
    // Check if running in a browser environment
    if (typeof window === 'undefined') return false;
    
    // Look for Electron-specific objects
    return !!(
      window.process?.type ||
      window.electron ||
      window.navigator.userAgent.includes('Electron')
    );
  };
  
  /**
   * Gets information about the current platform
   * 
   * @returns {Object} Platform information
   */
  export const getPlatformInfo = () => {
    if (isElectron()) {
      const userAgent = navigator.userAgent.toLowerCase();
      return {
        isElectron: true,
        isMac: userAgent.includes('mac'),
        isWindows: userAgent.includes('windows'),
        isLinux: userAgent.includes('linux'),
      };
    }
    
    return {
      isElectron: false,
      isMac: navigator.platform.includes('Mac'),
      isWindows: navigator.platform.includes('Win'),
      isLinux: navigator.platform.includes('Linux'),
    };
  };