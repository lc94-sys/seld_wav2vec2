#!/usr/bin/env python3
"""
Interactive CLI for RAG Chatbot Testing
Supports multiple users, sessions, and conversation management
"""

import yaml
import sys
from typing import Dict, List, Optional
from datetime import datetime
import json

# Import your retriever (adjust path as needed)
# Assuming notebook 03 code is converted to a module
try:
    from notebook_03_retrieval import HybridRetriever
except ImportError:
    print("⚠️  Could not import HybridRetriever. Make sure notebook 03 is accessible.")
    sys.exit(1)


class Colors:
    """ANSI color codes for pretty CLI output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class InteractiveCLI:
    """Interactive CLI for testing RAG system with multiple users"""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize CLI with configuration"""
        print(f"{Colors.CYAN}🚀 Initializing RAG Chatbot CLI...{Colors.ENDC}")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize retriever
        self.retriever = HybridRetriever(self.config)
        
        # User profiles (simulated users for testing)
        self.user_profiles = {
            '1': {
                'user_id': 'agent_001',
                'name': 'Alice (Support Agent)',
                'entitlement': 'agent_support',
                'org_id': 'org_123'
            },
            '2': {
                'user_id': 'agent_002',
                'name': 'Bob (Sales Agent)',
                'entitlement': 'agent_sales',
                'org_id': 'org_123'
            },
            '3': {
                'user_id': 'manager_001',
                'name': 'Carol (Manager)',
                'entitlement': 'agent_manager',
                'org_id': 'org_123'
            },
            '4': {
                'user_id': 'admin_001',
                'name': 'David (Admin)',
                'entitlement': 'agent_admin',
                'org_id': 'org_123'
            },
            'custom': {
                'user_id': 'custom_user',
                'name': 'Custom User',
                'entitlement': 'custom',
                'org_id': 'org_123'
            }
        }
        
        # Active sessions
        self.active_sessions: Dict[str, str] = {}  # user_id -> session_id
        
        print(f"{Colors.GREEN}✅ System initialized successfully!{Colors.ENDC}\n")
    
    def print_banner(self):
        """Print welcome banner"""
        banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║           🤖 RAG CHATBOT INTERACTIVE CLI 🤖              ║
║                                                           ║
║              Test with Multiple Users & Sessions          ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
{Colors.ENDC}
        """
        print(banner)
    
    def show_user_menu(self):
        """Show available users"""
        print(f"\n{Colors.YELLOW}{Colors.BOLD}👥 Available Users:{Colors.ENDC}")
        print(f"{Colors.YELLOW}{'─' * 60}{Colors.ENDC}")
        
        for key, profile in self.user_profiles.items():
            if key == 'custom':
                continue
            session_status = ""
            if profile['user_id'] in self.active_sessions:
                session_id = self.active_sessions[profile['user_id']]
                history_len = len(self.retriever.get_conversation_history(session_id))
                session_status = f" {Colors.GREEN}[Active Session: {history_len} messages]{Colors.ENDC}"
            
            print(f"  {Colors.BOLD}{key}.{Colors.ENDC} {profile['name']}")
            print(f"     Entitlement: {Colors.CYAN}{profile['entitlement']}{Colors.ENDC}{session_status}")
        
        print(f"\n  {Colors.BOLD}custom.{Colors.ENDC} Define custom user")
        print(f"  {Colors.BOLD}q.{Colors.ENDC} Quit")
        print(f"{Colors.YELLOW}{'─' * 60}{Colors.ENDC}")
    
    def select_user(self) -> Optional[Dict]:
        """Let user select which profile to use"""
        while True:
            choice = input(f"\n{Colors.BOLD}Select user (1-4, custom, q): {Colors.ENDC}").strip().lower()
            
            if choice == 'q':
                return None
            
            if choice == 'custom':
                return self.create_custom_user()
            
            if choice in self.user_profiles:
                return self.user_profiles[choice]
            
            print(f"{Colors.RED}❌ Invalid choice. Please try again.{Colors.ENDC}")
    
    def create_custom_user(self) -> Dict:
        """Create a custom user profile"""
        print(f"\n{Colors.CYAN}Creating custom user profile...{Colors.ENDC}")
        
        user_id = input("User ID: ").strip() or "custom_user"
        name = input("Name: ").strip() or "Custom User"
        entitlement = input("Entitlement: ").strip() or "agent_support"
        org_id = input("Organization ID [org_123]: ").strip() or "org_123"
        
        return {
            'user_id': user_id,
            'name': name,
            'entitlement': entitlement,
            'org_id': org_id
        }
    
    def get_or_create_session(self, user_profile: Dict) -> str:
        """Get existing session or create new one"""
        user_id = user_profile['user_id']
        
        # Check if user has active session
        if user_id in self.active_sessions:
            session_id = self.active_sessions[user_id]
            history = self.retriever.get_conversation_history(session_id)
            
            print(f"\n{Colors.GREEN}📝 Existing session found with {len(history)} messages{Colors.ENDC}")
            choice = input(f"{Colors.BOLD}Continue existing session? (y/n): {Colors.ENDC}").strip().lower()
            
            if choice == 'y':
                return session_id
            else:
                # Clear old session
                self.retriever.clear_session(session_id)
        
        # Create new session
        session_id = self.retriever.create_session(
            user_id=user_profile['user_id'],
            entitlement=user_profile['entitlement'],
            org_id=user_profile['org_id']
        )
        self.active_sessions[user_id] = session_id
        
        print(f"{Colors.GREEN}✅ New session created: {session_id[:8]}...{Colors.ENDC}")
        return session_id
    
    def show_chat_menu(self, user_profile: Dict):
        """Show options during chat"""
        print(f"\n{Colors.CYAN}{'─' * 60}{Colors.ENDC}")
        print(f"{Colors.BOLD}Commands:{Colors.ENDC}")
        print(f"  {Colors.CYAN}/history{Colors.ENDC}  - View conversation history")
        print(f"  {Colors.CYAN}/export{Colors.ENDC}   - Export session to file")
        print(f"  {Colors.CYAN}/clear{Colors.ENDC}    - Clear current session")
        print(f"  {Colors.CYAN}/switch{Colors.ENDC}   - Switch to different user")
        print(f"  {Colors.CYAN}/stats{Colors.ENDC}    - Show cache statistics")
        print(f"  {Colors.CYAN}/help{Colors.ENDC}     - Show this menu")
        print(f"  {Colors.CYAN}/quit{Colors.ENDC}     - Exit CLI")
        print(f"{Colors.CYAN}{'─' * 60}{Colors.ENDC}")
    
    def chat_session(self, user_profile: Dict, session_id: str):
        """Interactive chat session"""
        print(f"\n{Colors.GREEN}{Colors.BOLD}💬 Chat Session Started{Colors.ENDC}")
        print(f"{Colors.GREEN}User: {user_profile['name']} ({user_profile['entitlement']}){Colors.ENDC}")
        print(f"{Colors.GREEN}Session: {session_id[:8]}...{Colors.ENDC}")
        
        self.show_chat_menu(user_profile)
        
        while True:
            # Get user input
            query = input(f"\n{Colors.BOLD}{user_profile['name']}:{Colors.ENDC} ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.startswith('/'):
                if self.handle_command(query, user_profile, session_id):
                    break  # Exit chat if command returns True
                continue
            
            # Process query
            print(f"\n{Colors.CYAN}🔍 Searching...{Colors.ENDC}")
            
            try:
                result = self.retriever.query_with_session(
                    session_id=session_id,
                    query=query,
                    tags=None,  # Can be made configurable
                    top_k=5,
                    history_limit=2
                )
                
                # Display answer
                print(f"\n{Colors.GREEN}{Colors.BOLD}🤖 Assistant:{Colors.ENDC}")
                print(f"{Colors.GREEN}{result['answer']}{Colors.ENDC}")
                
                # Display sources
                if result.get('sources'):
                    print(f"\n{Colors.CYAN}{Colors.BOLD}📚 Sources:{Colors.ENDC}")
                    for i, source in enumerate(result['sources'][:3], 1):
                        print(f"  {i}. {source['title']} (score: {source['score']:.3f})")
                
                # Display timing if available
                if result.get('timings'):
                    total_time = result['timings'].get('total', 0)
                    print(f"\n{Colors.YELLOW}⏱️  Response time: {total_time:.2f}s{Colors.ENDC}")
                
            except Exception as e:
                print(f"\n{Colors.RED}❌ Error: {str(e)}{Colors.ENDC}")
    
    def handle_command(self, command: str, user_profile: Dict, session_id: str) -> bool:
        """Handle special commands. Returns True if should exit chat."""
        cmd = command.lower().strip()
        
        if cmd == '/quit':
            return True
        
        elif cmd == '/switch':
            return True
        
        elif cmd == '/help':
            self.show_chat_menu(user_profile)
        
        elif cmd == '/history':
            self.show_history(session_id)
        
        elif cmd == '/export':
            self.export_session(session_id)
        
        elif cmd == '/clear':
            self.clear_session(session_id, user_profile)
        
        elif cmd == '/stats':
            self.show_stats()
        
        else:
            print(f"{Colors.RED}❌ Unknown command: {command}{Colors.ENDC}")
            print(f"Type {Colors.CYAN}/help{Colors.ENDC} for available commands")
        
        return False
    
    def show_history(self, session_id: str):
        """Display conversation history"""
        history = self.retriever.get_conversation_history(session_id)
        
        if not history:
            print(f"\n{Colors.YELLOW}📝 No conversation history yet{Colors.ENDC}")
            return
        
        print(f"\n{Colors.CYAN}{Colors.BOLD}📜 Conversation History ({len(history)} messages):{Colors.ENDC}")
        print(f"{Colors.CYAN}{'═' * 60}{Colors.ENDC}")
        
        for i, turn in enumerate(history, 1):
            timestamp = turn.get('timestamp', 'N/A')
            print(f"\n{Colors.BOLD}Turn {i} [{timestamp}]{Colors.ENDC}")
            print(f"{Colors.BLUE}User:{Colors.ENDC} {turn['query']}")
            print(f"{Colors.GREEN}Assistant:{Colors.ENDC} {turn['answer'][:150]}...")
            
            if turn.get('sources'):
                print(f"{Colors.CYAN}Sources:{Colors.ENDC} {', '.join([s['title'] for s in turn['sources'][:2]])}")
        
        print(f"{Colors.CYAN}{'═' * 60}{Colors.ENDC}")
    
    def export_session(self, session_id: str):
        """Export session to JSON file"""
        filename = f"session_{session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            self.retriever.export_session(session_id, filename)
            print(f"\n{Colors.GREEN}✅ Session exported to: {filename}{Colors.ENDC}")
        except Exception as e:
            print(f"\n{Colors.RED}❌ Export failed: {str(e)}{Colors.ENDC}")
    
    def clear_session(self, session_id: str, user_profile: Dict):
        """Clear current session"""
        confirm = input(f"\n{Colors.YELLOW}⚠️  Clear all conversation history? (y/n): {Colors.ENDC}").strip().lower()
        
        if confirm == 'y':
            self.retriever.clear_session(session_id)
            
            # Create new session
            new_session_id = self.retriever.create_session(
                user_id=user_profile['user_id'],
                entitlement=user_profile['entitlement'],
                org_id=user_profile['org_id']
            )
            self.active_sessions[user_profile['user_id']] = new_session_id
            
            print(f"{Colors.GREEN}✅ Session cleared. New session created.{Colors.ENDC}")
        else:
            print(f"{Colors.YELLOW}Cancelled.{Colors.ENDC}")
    
    def show_stats(self):
        """Show system statistics"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}📊 System Statistics:{Colors.ENDC}")
        print(f"{Colors.CYAN}{'─' * 60}{Colors.ENDC}")
        
        # Cache stats (if OptimizedHybridRetriever)
        if hasattr(self.retriever, 'print_cache_stats'):
            self.retriever.print_cache_stats()
        
        # Active sessions
        print(f"\n{Colors.BOLD}Active Sessions:{Colors.ENDC} {len(self.active_sessions)}")
        for user_id, session_id in self.active_sessions.items():
            history = self.retriever.get_conversation_history(session_id)
            print(f"  • {user_id}: {len(history)} messages")
        
        print(f"{Colors.CYAN}{'─' * 60}{Colors.ENDC}")
    
    def run(self):
        """Main CLI loop"""
        self.print_banner()
        
        try:
            while True:
                self.show_user_menu()
                
                # Select user
                user_profile = self.select_user()
                if user_profile is None:
                    break
                
                # Get or create session
                session_id = self.get_or_create_session(user_profile)
                
                # Start chat
                self.chat_session(user_profile, session_id)
        
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}⚠️  Interrupted by user{Colors.ENDC}")
        
        finally:
            print(f"\n{Colors.CYAN}👋 Thanks for using RAG Chatbot CLI!{Colors.ENDC}")
            print(f"{Colors.CYAN}💾 All sessions are saved and can be resumed later.{Colors.ENDC}\n")


def main():
    """Entry point"""
    cli = InteractiveCLI()
    cli.run()


if __name__ == "__main__":
    main()
