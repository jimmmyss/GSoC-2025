# Discord Bot

## Overview
Interactive Discord bot with button-based UI for server administration. Provides announcement posting, message clearing, and automated channel management.

### Announce Button
**Purpose**: Send messages to any specified channel

**Workflow**:
1. User clicks "Announce" button
2. Bot prompts: "Mention the channel you want to announce in"
3. User types channel mention (e.g., `#announcements`)
4. Bot validates channel mention
5. Bot prompts: "Write the message you want to announce"
6. User types message content
7. Bot sends message to specified channel
8. Returns to main menu

### Clear Button
**Purpose**: Delete specified number of messages from channels

**Workflow**:
1. User clicks "Clear" button
2. Bot prompts: "Mention the channel you want to erase messages"
3. User types channel mention (e.g., `#general`)
4. Bot validates channel mention
5. Bot prompts: "Write the number of messages you want to clear"
6. User enters numeric value
7. Bot validates numeric input
8. Bot deletes specified number of messages
9. Returns to main menu

### Help Button
**Purpose**: Display information about available commands

**Workflow**:
1. User clicks "Help" button
2. Bot displays embed with command descriptions:
   - **Announce**: "Announces a message to a specified channel"
   - **Clear**: "Clears messages from the channel"
3. "Back" button returns to main menu

### Create Invite Command
**Purpose**: Generate permanent server invites

**Usage**: Type `.createinvite` in any channel
**Functionality**:
- Creates permanent invite link (no expiration, unlimited uses)
- Uses specific channel ID for invite generation
- Responds with clickable invite URL

## Installation & Setup

### Prerequisites
```bash
pip install discord.py
```

### Configuration
Replace bot token in line 225:
```python
bot.run('YOUR_BOT_TOKEN_HERE')
```

### How to run the bot
```bash
python main.py
```