#!/usr/bin/env python3
"""
Simple Chat System Models
"""
import sqlite3
import os
from datetime import datetime

def get_db_connection():
    """Get database connection using the same path resolution as main app."""
    # Use the same database path resolution logic as database.py
    default_local_db = 'anemia_classification.db'
    env_db_path = os.environ.get('DATABASE_PATH')
    volume_mount = os.environ.get('RAILWAY_VOLUME_MOUNT_PATH')
    running_on_railway = any(k in os.environ for k in [
        'RAILWAY_PROJECT_ID', 'RAILWAY_ENVIRONMENT', 'RAILWAY_STATIC_URL', 'RAILWAY_GIT_COMMIT_SHA'
    ])

    if env_db_path:
        db_path = env_db_path
    elif volume_mount:
        db_path = os.path.join(volume_mount, 'anemocheck', default_local_db)
    elif running_on_railway:
        # Common default mount path for Railway volumes
        db_path = os.path.join('/data', 'anemocheck', default_local_db)
    else:
        db_path = default_local_db

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_chat_tables():
    """Initialize chat tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create conversations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            admin_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (admin_id) REFERENCES users (id)
        )
    ''')
    
    # Create messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            sender_id INTEGER NOT NULL,
            message_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES chat_conversations (id),
            FOREIGN KEY (sender_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Simple chat tables initialized successfully.")

def create_conversation(user_id, admin_id=None):
    """Create a new conversation."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Use Philippines time for timestamp
        from timezone_utils import get_philippines_time_for_db
        ph_timestamp = get_philippines_time_for_db()
        
        cursor.execute('''
            INSERT INTO chat_conversations (user_id, admin_id, created_at)
            VALUES (?, ?, ?)
        ''', (user_id, admin_id, ph_timestamp))
        
        conversation_id = cursor.lastrowid
        conn.commit()
        return True, conversation_id
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def send_message(conversation_id, sender_id, message_text):
    """Send a message."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Use Philippines time for timestamp
        from timezone_utils import get_philippines_time_for_db
        ph_timestamp = get_philippines_time_for_db()
        
        cursor.execute('''
            INSERT INTO chat_messages (conversation_id, sender_id, message_text, created_at)
            VALUES (?, ?, ?, ?)
        ''', (conversation_id, sender_id, message_text, ph_timestamp))
        
        message_id = cursor.lastrowid
        conn.commit()
        return True, message_id
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def get_conversation_messages(conversation_id):
    """Get messages for a conversation."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT m.*, u.username, u.first_name, u.last_name
        FROM chat_messages m
        JOIN users u ON m.sender_id = u.id
        WHERE m.conversation_id = ?
        ORDER BY m.created_at ASC
    ''', (conversation_id,))
    
    messages = []
    for row in cursor.fetchall():
        messages.append({
            'id': row['id'],
            'sender_id': row['sender_id'],
            'message_text': row['message_text'],
            'created_at': row['created_at'],
            'username': row['username'],
            'first_name': row['first_name'],
            'last_name': row['last_name']
        })
    
    conn.close()
    return messages

def get_user_conversations(user_id, is_admin=False):
    """Get conversations for a user with last message info."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if is_admin:
        cursor.execute('''
            SELECT c.*, u.username, u.first_name, u.last_name,
                   (SELECT COUNT(*) FROM chat_messages WHERE conversation_id = c.id) as message_count,
                   (SELECT message_text FROM chat_messages WHERE conversation_id = c.id ORDER BY created_at DESC LIMIT 1) as last_message,
                   (SELECT created_at FROM chat_messages WHERE conversation_id = c.id ORDER BY created_at DESC LIMIT 1) as last_message_time
            FROM chat_conversations c
            JOIN users u ON c.user_id = u.id
            WHERE c.admin_id = ?
            ORDER BY COALESCE((SELECT created_at FROM chat_messages WHERE conversation_id = c.id ORDER BY created_at DESC LIMIT 1), c.created_at) DESC
        ''', (user_id,))
    else:
        cursor.execute('''
            SELECT c.*, u.username, u.first_name, u.last_name,
                   (SELECT COUNT(*) FROM chat_messages WHERE conversation_id = c.id) as message_count,
                   (SELECT message_text FROM chat_messages WHERE conversation_id = c.id ORDER BY created_at DESC LIMIT 1) as last_message,
                   (SELECT created_at FROM chat_messages WHERE conversation_id = c.id ORDER BY created_at DESC LIMIT 1) as last_message_time
            FROM chat_conversations c
            LEFT JOIN users u ON c.admin_id = u.id
            WHERE c.user_id = ?
            ORDER BY COALESCE((SELECT created_at FROM chat_messages WHERE conversation_id = c.id ORDER BY created_at DESC LIMIT 1), c.created_at) DESC
        ''', (user_id,))
    
    conversations = []
    for row in cursor.fetchall():
        conversations.append({
            'id': row['id'],
            'user_id': row['user_id'],
            'admin_id': row['admin_id'],
            'created_at': row['created_at'],
            'username': row['username'],
            'first_name': row['first_name'],
            'last_name': row['last_name'],
            'message_count': row['message_count'],
            'last_message': row['last_message'],
            'last_message_time': row['last_message_time']
        })
    
    conn.close()
    return conversations

def get_all_users():
    """Get all users."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, first_name, last_name, email, is_admin
        FROM users
        ORDER BY username
    ''')
    
    users = []
    for row in cursor.fetchall():
        users.append({
            'id': row['id'],
            'username': row['username'],
            'first_name': row['first_name'],
            'last_name': row['last_name'],
            'email': row['email'],
            'is_admin': row['is_admin']
        })
    
    conn.close()
    return users
