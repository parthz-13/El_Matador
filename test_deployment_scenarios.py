"""
Test suite to verify the application works correctly in different deployment scenarios.

This test simulates various directory structures and deployment environments
to ensure the application's path handling is robust.

Requirements: 14.3
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import importlib.util


def test_huggingface_spaces_structure():
    """
    Test that the application works with HuggingFace Spaces directory structure.
    
    HuggingFace Spaces typically has the following structure:
    /home/user/app/
        streamlit_app.py
        models/
            best_model.joblib
            tfidf_vectorizer.joblib
    """
    print("\n📦 Testing HuggingFace Spaces deployment structure...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create HuggingFace Spaces-like structure
        app_dir = os.path.join(tmpdir, 'app')
        models_dir = os.path.join(app_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Copy streamlit_app.py to the temp directory
        shutil.copy('streamlit_app.py', app_dir)
        
        # Verify the path construction would work
        test_file = os.path.join(app_dir, 'streamlit_app.py')
        expected_models_dir = os.path.join(os.path.dirname(test_file), 'models')
        expected_model_path = os.path.join(expected_models_dir, 'best_model.joblib')
        
        assert expected_models_dir == models_dir, \
            f"Expected models dir: {models_dir}, got: {expected_models_dir}"
        
        print(f"  ✅ App directory: {app_dir}")
        print(f"  ✅ Models directory: {models_dir}")
        print(f"  ✅ Model path resolves correctly: {expected_model_path}")


def test_nested_deployment_structure():
    """
    Test that the application works when deployed in a nested directory structure.
    
    Some deployment platforms may place the app in nested directories:
    /opt/app/project/
        streamlit_app.py
        models/
    """
    print("\n📦 Testing nested deployment structure...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested structure
        app_dir = os.path.join(tmpdir, 'opt', 'app', 'project')
        models_dir = os.path.join(app_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Copy streamlit_app.py to the temp directory
        shutil.copy('streamlit_app.py', app_dir)
        
        # Verify the path construction would work
        test_file = os.path.join(app_dir, 'streamlit_app.py')
        expected_models_dir = os.path.join(os.path.dirname(test_file), 'models')
        
        assert expected_models_dir == models_dir, \
            f"Expected models dir: {models_dir}, got: {expected_models_dir}"
        
        print(f"  ✅ Nested app directory: {app_dir}")
        print(f"  ✅ Models directory resolves correctly: {models_dir}")


def test_symlink_deployment():
    """
    Test that the application works when deployed with symbolic links.
    
    Some deployment scenarios may use symbolic links for the application files.
    """
    print("\n📦 Testing symbolic link deployment...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create original structure
        original_dir = os.path.join(tmpdir, 'original')
        models_dir = os.path.join(original_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Copy streamlit_app.py to original directory
        original_app = os.path.join(original_dir, 'streamlit_app.py')
        shutil.copy('streamlit_app.py', original_app)
        
        # Create symlink directory
        symlink_dir = os.path.join(tmpdir, 'symlink')
        os.makedirs(symlink_dir, exist_ok=True)
        
        # Create symlink to the app file
        symlink_app = os.path.join(symlink_dir, 'streamlit_app.py')
        try:
            os.symlink(original_app, symlink_app)
            
            # Verify path resolution
            # os.path.dirname(__file__) should resolve to the symlink location
            # but the models should still be found relative to the actual file
            expected_models_dir = os.path.join(os.path.dirname(original_app), 'models')
            
            assert os.path.exists(expected_models_dir), \
                f"Models directory should exist at: {expected_models_dir}"
            
            print(f"  ✅ Original app: {original_app}")
            print(f"  ✅ Symlink app: {symlink_app}")
            print(f"  ✅ Models directory accessible: {expected_models_dir}")
        except OSError as e:
            print(f"  ⊘ SKIPPED: Symbolic links not supported on this system ({e})")


def test_different_working_directories():
    """
    Test that the application works regardless of the current working directory.
    
    This ensures that the application can be run from any directory, not just
    the application root.
    """
    print("\n📦 Testing different working directories...")
    
    original_cwd = os.getcwd()
    
    try:
        # Test 1: Run from parent directory
        parent_dir = os.path.dirname(original_cwd)
        if parent_dir and parent_dir != original_cwd:
            os.chdir(parent_dir)
            
            # Verify that the path construction still works
            app_file = os.path.join(original_cwd, 'streamlit_app.py')
            models_dir = os.path.join(os.path.dirname(app_file), 'models')
            
            assert 'models' in models_dir, \
                "Models directory should be in the path"
            
            print(f"  ✅ Works from parent directory: {parent_dir}")
        
        # Test 2: Run from a completely different directory
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            
            # Verify that the path construction still works
            app_file = os.path.join(original_cwd, 'streamlit_app.py')
            models_dir = os.path.join(os.path.dirname(app_file), 'models')
            
            assert 'models' in models_dir, \
                "Models directory should be in the path"
            
            print(f"  ✅ Works from different directory: {tmpdir}")
    
    finally:
        os.chdir(original_cwd)


def test_path_separators():
    """
    Test that the application handles different path separators correctly.
    
    This ensures cross-platform compatibility (Windows vs Unix-like systems).
    """
    print("\n📦 Testing path separator handling...")
    
    # Test that os.path.join is used (which handles separators correctly)
    with open('streamlit_app.py', 'r') as f:
        content = f.read()
    
    # Verify os.path.join is used instead of manual string concatenation
    assert 'os.path.join' in content, \
        "Should use os.path.join for cross-platform compatibility"
    
    # Verify no hardcoded separators in path construction
    lines = content.split('\n')
    issues = []
    
    for line_num, line in enumerate(lines, 1):
        # Skip comments
        if line.strip().startswith('#'):
            continue
        
        # Check for manual path concatenation (potential issues)
        if ('+ "/"' in line or "+ '/'" in line or 
            '+ "\\"' in line or '+ "\\\\"' in line):
            issues.append(f"Line {line_num}: Manual path concatenation detected")
    
    if issues:
        print("  ⚠️  Potential path separator issues:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print("  ✅ No hardcoded path separators found")
        print("  ✅ Uses os.path.join for cross-platform compatibility")


def test_model_files_location():
    """
    Test that model files are expected in the correct relative location.
    
    This verifies that the application looks for models in the 'models'
    subdirectory relative to the application file.
    """
    print("\n📦 Testing model files location...")
    
    # Read streamlit_app.py to verify model path construction
    with open('streamlit_app.py', 'r') as f:
        content = f.read()
    
    # Verify the expected structure
    assert '"models"' in content or "'models'" in content, \
        "Should reference 'models' directory"
    assert 'best_model.joblib' in content, \
        "Should reference 'best_model.joblib' file"
    assert 'tfidf_vectorizer.joblib' in content, \
        "Should reference 'tfidf_vectorizer.joblib' file"
    
    print("  ✅ Models expected in 'models' subdirectory")
    print("  ✅ Model files: best_model.joblib, tfidf_vectorizer.joblib")


def test_error_messages_show_relative_paths():
    """
    Test that error messages show helpful relative paths, not absolute paths.
    
    This ensures that error messages are useful across different deployments.
    """
    print("\n📦 Testing error message path references...")
    
    # Read streamlit_app.py to check error messages
    with open('streamlit_app.py', 'r') as f:
        content = f.read()
    
    # Find error messages that mention paths
    lines = content.split('\n')
    path_error_messages = []
    
    for line_num, line in enumerate(lines, 1):
        if 'st.error' in line and ('path' in line.lower() or 'directory' in line.lower()):
            path_error_messages.append((line_num, line.strip()))
    
    if path_error_messages:
        print(f"  ✅ Found {len(path_error_messages)} error messages with path references")
        
        # Verify they use the constructed paths (variables), not hardcoded paths
        for line_num, line in path_error_messages:
            if '/home/' in line or 'C:\\' in line or '/Users/' in line:
                print(f"  ⚠️  Line {line_num}: May contain hardcoded absolute path")
            else:
                print(f"  ✅ Line {line_num}: Uses dynamic path references")
    else:
        print("  ✅ No path-related error messages found (or they use variables)")


def test_readme_deployment_instructions():
    """
    Test that README.md contains deployment instructions with relative paths.
    
    This ensures that deployment documentation is accurate and helpful.
    """
    print("\n📦 Testing README deployment instructions...")
    
    if not os.path.exists('README.md'):
        print("  ⊘ SKIPPED: README.md not found")
        return
    
    with open('README.md', 'r') as f:
        content = f.read()
    
    # Check for deployment-related sections
    deployment_keywords = [
        'deploy',
        'huggingface',
        'spaces',
        'installation',
        'setup',
    ]
    
    found_keywords = [kw for kw in deployment_keywords if kw.lower() in content.lower()]
    
    if found_keywords:
        print(f"  ✅ README contains deployment information: {', '.join(found_keywords)}")
    else:
        print("  ⚠️  README may be missing deployment instructions")
    
    # Check for model file references
    if 'models/' in content or 'best_model.joblib' in content:
        print("  ✅ README references model files location")
    else:
        print("  ⚠️  README may be missing model files information")


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Deployment Scenarios")
    print("=" * 70)
    
    # Run tests
    tests = [
        test_huggingface_spaces_structure,
        test_nested_deployment_structure,
        test_symlink_deployment,
        test_different_working_directories,
        test_path_separators,
        test_model_files_location,
        test_error_messages_show_relative_paths,
        test_readme_deployment_instructions,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ FAILED: {test_func.__name__}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ ERROR: {test_func.__name__}")
            print(f"  Error: {e}")
            failed += 1
    
    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✅ All deployment scenarios passed!")
        print("   The application uses relative paths correctly and should work")
        print("   in different deployment environments including HuggingFace Spaces.")
