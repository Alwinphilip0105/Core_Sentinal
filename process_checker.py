"""
Process checking utilities for CoreSentinal.
Checks if the application is already running and prevents multiple instances.
"""

import os
import sys
import psutil
from typing import List, Optional


def get_current_process_name() -> str:
    """Get the name of the current Python process."""
    return os.path.basename(sys.executable)


def get_running_python_processes() -> List[dict]:
    """
    Get all running Python processes.
    
    Returns:
        List of dicts with process info: {pid, name, cmdline, memory_mb}
    """
    processes = []
    current_pid = os.getpid()
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            info = proc.info
            if 'python' in info['name'].lower() or 'pythonw' in info['name'].lower():
                cmdline = ' '.join(info['cmdline']) if info['cmdline'] else ''
                memory_mb = info['memory_info'].rss / (1024 * 1024) if info['memory_info'] else 0
                
                processes.append({
                    'pid': info['pid'],
                    'name': info['name'],
                    'cmdline': cmdline,
                    'memory_mb': round(memory_mb, 2),
                    'is_current': info['pid'] == current_pid
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    return processes


def find_core_sentinal_processes() -> List[dict]:
    """
    Find all running CoreSentinal processes.
    
    Returns:
        List of CoreSentinal process info dicts
    """
    all_processes = get_running_python_processes()
    core_sentinal_processes = []
    
    for proc in all_processes:
        cmdline_lower = proc['cmdline'].lower()
        # Check if this is a CoreSentinal process
        if 'coresentinal' in cmdline_lower or 'main.py' in cmdline_lower:
            if 'coresentinal' in cmdline_lower or 'main.py' in cmdline_lower:
                core_sentinal_processes.append(proc)
    
    return core_sentinal_processes


def is_already_running(exclude_current: bool = True) -> bool:
    """
    Check if CoreSentinal is already running.
    
    Args:
        exclude_current: If True, exclude the current process from check
        
    Returns:
        True if another instance is running, False otherwise
    """
    processes = find_core_sentinal_processes()
    current_pid = os.getpid()
    
    if exclude_current:
        # Check if any process other than current is running
        return any(proc['pid'] != current_pid for proc in processes)
    else:
        # Check if any process is running (including current)
        return len(processes) > 0


def print_running_processes():
    """Print all running Python processes for debugging."""
    processes = get_running_python_processes()
    
    print("\n=== Running Python Processes ===")
    print(f"{'PID':<8} {'Name':<15} {'Memory (MB)':<12} {'Command Line'}")
    print("-" * 80)
    
    for proc in processes:
        marker = " [CURRENT]" if proc['is_current'] else ""
        cmdline = proc['cmdline'][:60] + "..." if len(proc['cmdline']) > 60 else proc['cmdline']
        print(f"{proc['pid']:<8} {proc['name']:<15} {proc['memory_mb']:<12} {cmdline}{marker}")
    
    print(f"\nTotal: {len(processes)} Python process(es)")
    print()


def print_core_sentinal_status():
    """Print CoreSentinal-specific process status."""
    processes = find_core_sentinal_processes()
    current_pid = os.getpid()
    
    print("\n=== CoreSentinal Process Status ===")
    
    if not processes:
        print("No CoreSentinal processes found.")
        return
    
    for proc in processes:
        status = "CURRENT" if proc['pid'] == current_pid else "RUNNING"
        print(f"PID: {proc['pid']} | Status: {status} | Memory: {proc['memory_mb']} MB")
        print(f"  Command: {proc['cmdline']}")
    
    other_instances = [p for p in processes if p['pid'] != current_pid]
    if other_instances:
        print(f"\n⚠️  WARNING: {len(other_instances)} other instance(s) detected!")
    else:
        print("\n✓ Only this instance is running.")
    
    print()


if __name__ == "__main__":
    # Test the process checker
    print("Current Process PID:", os.getpid())
    print_core_sentinal_status()
    print_running_processes()
