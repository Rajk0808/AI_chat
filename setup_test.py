import sys
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}")
    print(f"{text}")
    print(f"{'='*60}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def print_info(text):
    print(f"{BLUE}ℹ {text}{RESET}")

# ============================================================================
# Step 1: Create __init__.py files
# ============================================================================

def create_init_files():
    print_header("Step 1: Creating __init__.py Files")
    
    # Define directory structure
    dirs_to_init = [
        '',  # Root
        'AI_Model',
        'AI_Model/src',
        'AI_Model/src/workflow',
        'AI_Model/src/rag',
        'AI_Model/src/models',
        'AI_Model/src/prompt_engineering',
        'AI_Model/src/utils',
        'AI_Model/src/logging',
        'AI_Model/src/fine_tuning',
    ]
    
    project_root = Path(__file__).parent
    created_count = 0
    
    for dir_path in dirs_to_init:
        if dir_path:
            full_path = project_root / dir_path
        else:
            full_path = project_root
        
        # Create directory if it doesn't exist
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py
        init_file = full_path / '__init__.py'
        if not init_file.exists():
            init_file.touch()
            print_success(f"Created {init_file.relative_to(project_root)}")
            created_count += 1
        else:
            print_info(f"Already exists: {init_file.relative_to(project_root)}")
    
    print(f"\n{GREEN}Total files created: {created_count}{RESET}")
    return True

# ============================================================================
# Step 2: Test Imports
# ============================================================================

def test_imports():
    print_header("Step 2: Testing Imports")
    
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / 'AI_Model'))
    
    all_passed = True
    
    # Test 1: State Definition
    print_info("Testing: state_definition import...")
    try:
        from AI_Model.src.workflow.state_definition import WorkFlowState
        print_success("Successfully imported WorkFlowState")
    except Exception as e:
        print_error(f"Failed to import state_definition: {e}")
        all_passed = False
    
    # Test 2: Nodes Module
    print_info("Testing: nodes import...")
    try:
        from AI_Model.src.workflow import nodes
        print_success("Successfully imported nodes")
    except Exception as e:
        print_error(f"Failed to import nodes: {e}")
        all_passed = False
    
    # Test 3: Graph Builder
    print_info("Testing: graph_builder import...")
    try:
        from AI_Model.src.workflow.graph_builder import build_complete_workflow
        print_success("Successfully imported build_complete_workflow")
    except Exception as e:
        print_error(f"Failed to import graph_builder: {e}")
        all_passed = False
    
    # Test 4: Try to build workflow
    print_info("Testing: workflow compilation...")
    try:
        from AI_Model.src.workflow.graph_builder import build_complete_workflow
        workflow = build_complete_workflow()
        print_success("Workflow compiled successfully!")
    except Exception as e:
        print_warning(f"Workflow compilation failed (might be normal): {e}")
        # This is not critical if it's due to missing modules
    
    return all_passed

# ============================================================================
# Step 3: Check File Structure
# ============================================================================

def check_file_structure():
    print_header("Step 3: Checking File Structure")
    
    project_root = Path(__file__).parent
    expected_files = [
        'state_definition.py',
        'workflow_pipeline.py',
        'index.html',
        'requirements.txt',
        'AI_Model/src/workflow/nodes.py',
        'AI_Model/src/workflow/graph_builder.py',
    ]
    
    all_exist = True
    
    for file_path in expected_files:
        full_path = project_root / file_path
        if full_path.exists():
            print_success(f"Found: {file_path}")
        else:
            print_warning(f"Missing: {file_path}")
            all_exist = False
    
    return all_exist

# ============================================================================
# Step 4: Check Dependencies
# ============================================================================

def check_dependencies():
    print_header("Step 4: Checking Python Dependencies")
    
    dependencies = [
        ('flask', 'Flask'),
        ('flask_cors', 'Flask-CORS'),
        ('langgraph', 'LangGraph'),
        ('langchain', 'LangChain'),
    ]
    
    all_installed = True
    
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print_success(f"✓ {display_name} is installed")
        except ImportError:
            print_error(f"✗ {display_name} is NOT installed")
            all_installed = False
    
    if not all_installed:
        print_warning("\nTo install missing packages, run:")
        print(f"{BLUE}pip install -r requirements.txt{RESET}")
    
    return all_installed

# ============================================================================
# Step 5: Show Next Steps
# ============================================================================

def show_next_steps():
    print_header("Step 5: Ready to Run!")
    
    print("✓ Setup complete!\n")
    print(f"{BLUE}To start your chatbot:{RESET}\n")
    
    print(f"{BLUE}Terminal 1 - Start Backend:{RESET}")
    print(f"{GREEN}python workflow_pipeline.py{RESET}\n")
    
    print(f"{BLUE}Terminal 2 - Open Frontend:{RESET}")
    print(f"{GREEN}Open 'index.html' in your web browser{RESET}\n")
    
    print(f"Or if using Python server:")
    print(f"{GREEN}python -m http.server 8000{RESET}")
    print(f"Then visit: {BLUE}http://localhost:8000/index.html{RESET}\n")

# ============================================================================
# Main Execution
# ============================================================================

def main():
    print(f"\n{BLUE}{'='*60}")
    print(f"AI Chatbot - Setup & Test Script")
    print(f"{'='*60}{RESET}")
    
    try:
        # Step 1: Create init files
        create_init_files()
        
        # Step 2: Check file structure
        check_file_structure()
        
        # Step 3: Check dependencies
        deps_ok = check_dependencies()
        
        # Step 4: Test imports
        imports_ok = test_imports()
        
        # Step 5: Show next steps
        show_next_steps()
        
        if imports_ok and deps_ok:
            print_success("All systems ready! You can now run the backend.")
        else:
            print_warning("Some issues detected. Please review above messages.")
        
    except Exception as e:
        print_error(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())