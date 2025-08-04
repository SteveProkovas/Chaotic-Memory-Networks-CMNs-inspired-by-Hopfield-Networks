
"""
Colab Experiment Runner - Executes Chaotic Memory Network experiments on Google Colab
Handles environment setup, experiment execution, and result management
"""

import sys
import os
import importlib
import argparse
import traceback
from colab.colab_utils import mount_drive, setup_project_dir, prevent_timeout, save_session_state

def run_experiment(experiment_name, params=None):
    """
    Dynamically loads and executes an experiment module
    
    Args:
        experiment_name: Name of the experiment module (without .py extension)
        params: Dictionary of parameters to pass to the experiment
        
    Returns:
        True if experiment executed successfully, False otherwise
    """
    try:
        # Construct the full module path (experiments.colab.<experiment_name>)
        module_path = f"experiments.colab.{experiment_name}"
        
        # Import the experiment module dynamically
        experiment_module = importlib.import_module(module_path)
        
        # Check if the module has a 'main' function
        if not hasattr(experiment_module, 'main'):
            print(f"⚠️ Error: Module '{module_path}' has no main() function")
            return False
        
        print(f"🚀 Starting experiment: {experiment_name}")
        
        # Execute the experiment's main function with parameters
        success = experiment_module.main(params or {})
        
        if success:
            print(f"✅ Experiment '{experiment_name}' completed successfully")
        else:
            print(f"❌ Experiment '{experiment_name}' reported failure")
        
        return success
        
    except ImportError:
        print(f"🔥 Import error: Could not find experiment module '{module_path}'")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"💥 Critical error during experiment execution: {str(e)}")
        traceback.print_exc()
        return False

def parse_params(param_list):
    """
    Parses command-line parameters in key=value format
    
    Args:
        param_list: List of strings in 'key=value' format
        
    Returns:
        Dictionary of parsed parameters
    """
    params = {}
    if not param_list:
        return params
        
    for param in param_list:
        # Split only on the first '=' to handle values containing equals
        if '=' in param:
            key, value = param.split('=', 1)
            params[key.strip()] = value.strip()
        else:
            print(f"⚠️ Warning: Ignoring malformed parameter '{param}'")
            
    return params

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description='Colab Experiment Runner - Executes Chaotic Memory Network experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('experiment', 
                        type=str, 
                        help='Name of experiment module to run (without .py extension)')
    
    # Optional arguments
    parser.add_argument('--params', 
                        nargs='*', 
                        help='Key=value parameters to pass to the experiment')
    
    parser.add_argument('--no-timeout', 
                        action='store_true',
                        help='Disable Colab timeout prevention mechanism')
    
    parser.add_argument('--drive-mount', 
                        default='/content/drive',
                        help='Mount point for Google Drive')
    
    parser.add_argument('--project-dir', 
                        default='Chaotic_Memory_Networks',
                        help='Project directory name in Google Drive')
    
    parser.add_argument('--save-interval', 
                        type=int, 
                        default=30,
                        help='Session save interval in minutes (0 to disable)')
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    print("="*60)
    print(f"⚙️ Starting Colab Experiment Runner")
    print(f"🔬 Experiment: {args.experiment}")
    print(f"📂 Project Directory: {args.project_dir}")
    print("="*60)
    
    # Step 1: Mount Google Drive
    print("\n🔗 Mounting Google Drive...")
    if mount_drive(args.drive_mount):
        print(f"✅ Drive mounted at {args.drive_mount}")
    else:
        print("❌ Drive mount failed! Exiting.")
        sys.exit(1)
    
    # Step 2: Set up project directory
    print("\n📁 Setting up project directory...")
    project_path = setup_project_dir(args.project_dir)
    print(f"✅ Project directory: {project_path}")
    os.chdir(project_path)
    print(f"📁 Current working directory: {os.getcwd()}")
    
    # Step 3: Prevent Colab timeouts (unless disabled)
    if not args.no_timeout:
        print("\n⏳ Enabling Colab timeout prevention...")
        prevent_timeout()
        print("✅ Session will refresh every 30 minutes to prevent disconnects")
    else:
        print("\n⚠️ Timeout prevention disabled - session may disconnect")
    
    # Step 4: Set up periodic session saving
    if args.save_interval > 0:
        print(f"\n💾 Will save session state every {args.save_interval} minutes")
        # This would be implemented with background threading in practice
        # For simplicity, we'll save at start and end in this script
        save_session_state()
    
    # Step 5: Parse experiment parameters
    print("\n🔍 Parsing experiment parameters...")
    params = parse_params(args.params)
    if params:
        print("📋 Parameters:")
        for key, value in params.items():
            print(f"  - {key}: {value}")
    else:
        print("📋 No parameters provided")
    
    # Step 6: Execute the experiment
    print("\n" + "="*60)
    print("🚀 Launching experiment...")
    print("="*60 + "\n")
    
    success = run_experiment(args.experiment, params)
    
    # Step 7: Final session save
    if args.save_interval > 0:
        print("\n💾 Saving final session state...")
        save_session_state()
    
    # Step 8: Exit with appropriate status
    print("\n" + "="*60)
    print(f"🏁 Experiment runner completed")
    print(f"📊 Result: {'SUCCESS' if success else 'FAILURE'}")
    print("="*60)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
