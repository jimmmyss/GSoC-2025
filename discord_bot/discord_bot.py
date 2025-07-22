import discord
from discord.ext import commands
import asyncio

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix=".", intents=intents)

class Buttons(discord.ui.View):
        
    @discord.ui.button(label="Announce", style=discord.ButtonStyle.primary)
    async def announce_callback(self, interaction: discord.Interaction, button: discord.ui.Button):
        cancel_flag = False

        view = discord.ui.View()
        back_button = discord.ui.Button(label="Back", style=discord.ButtonStyle.secondary)
        view.add_item(back_button)
        
        async def back_callback(interaction: discord.Interaction):
            nonlocal cancel_flag
            cancel_flag = True
            await interaction.response.edit_message(embed=discord.Embed(title="Commands",description="Please choose one of the commands below by clicking the buttons.",color=discord.Color.orange()),view=Buttons())

        back_button.callback = back_callback

        await interaction.response.edit_message(embed=discord.Embed(title="Announce", description="Mention the channel you want to announce in (e.g., #announcements):", color=discord.Color.orange()), view=view)

        def check(m):
            return m.channel == interaction.channel and m.author == interaction.user

        while not cancel_flag:
            message = await bot.wait_for('message', check=check)
            if cancel_flag:
                return
            if message.channel_mentions:
                break
            else:
                await interaction.followup.edit_message(message_id=interaction.message.id, embed=discord.Embed(title="Announce", description="No channel mention found. Please mention a channel (e.g., #announcements):", color=discord.Color.red()), view=view)
                
        channel = message.channel_mentions[0]
        await interaction.followup.edit_message(message_id=interaction.message.id, embed=discord.Embed(title="Announcing in " + channel.mention, description="Write the message you want to announce:", color=discord.Color.orange()), view=view)

        announcement_message = await bot.wait_for('message', check=check)
        if cancel_flag:
            return
        
        await interaction.followup.edit_message(message_id=interaction.message.id, embed=discord.Embed(title="Announced in " + channel.mention, description="Message sent!", color=discord.Color.green()), view=discord.ui.View())
        
        await channel.send(announcement_message.content)
        
        await asyncio.sleep(2)
        
        await interaction.followup.edit_message(message_id=interaction.message.id, embed=discord.Embed(title="Commands",description="Please choose one of the commands below by clicking the buttons.",color=discord.Color.orange()),view=Buttons())

        
    @discord.ui.button(label="Clear", style=discord.ButtonStyle.primary)
    async def clear_callback(self, interaction: discord.Interaction, button: discord.ui.Button):
        cancel_flag = False

        view = discord.ui.View()
        back_button = discord.ui.Button(label="Back", style=discord.ButtonStyle.secondary)
        view.add_item(back_button)
        
        async def back_callback(interaction: discord.Interaction):
            nonlocal cancel_flag
            cancel_flag = True
            await interaction.response.edit_message(embed=discord.Embed(title="Commands",description="Please choose one of the commands below by clicking the buttons.",color=discord.Color.orange()),view=Buttons())

        back_button.callback = back_callback

        await interaction.response.edit_message(embed=discord.Embed(title="Clear", description="Mention the channel you want to erase messages (e.g., #announcements):", color=discord.Color.orange()), view=view)

        def check(m):
            return m.channel == interaction.channel and m.author == interaction.user

        while not cancel_flag:
            message = await bot.wait_for('message', check=check)
            if cancel_flag:
                return
            if message.channel_mentions:
                break
            else:
                await interaction.followup.edit_message(message_id=interaction.message.id, embed=discord.Embed(title="Clear", description="No channel mention found. Please mention a channel (e.g., #announcements):", color=discord.Color.red()), view=view)
                
        channel = message.channel_mentions[0]
        await interaction.followup.edit_message(message_id=interaction.message.id, embed=discord.Embed(title="Clearing in " + channel.mention, description="Write the number of messages you want to clear:", color=discord.Color.orange()), view=view)

        while not cancel_flag:
            message = await bot.wait_for('message', check=check)
            if cancel_flag:
                return
            if all(char in "0123456789" for char in message.content):
                break
            else:
                await interaction.followup.edit_message(message_id=interaction.message.id, embed=discord.Embed(title="Clear", description="Enter a valid number", color=discord.Color.red()), view=view)

        await interaction.followup.edit_message(message_id=interaction.message.id, embed=discord.Embed(title="Cleared in " + channel.mention, description=message.content + " messages deleted!", color=discord.Color.green()), view=discord.ui.View())
       
        await channel.purge(limit=int(message.content))

        await asyncio.sleep(2)
        
        await interaction.followup.edit_message(message_id=interaction.message.id, embed=discord.Embed(title="Commands",description="Please choose one of the commands below by clicking the buttons.",color=discord.Color.orange()),view=Buttons())

        # clearall_button = discord.ui.Button(label="All", style=discord.ButtonStyle.red)
        # view.add_item(clearall_button)
        # async def clearall_callback(interaction: discord.Interaction):
        #     await interaction.response.edit_message(embed=discord.Embed(title="Commands",description="Please choose one of the commands below by clicking the buttons.",color=discord.Color.orange()),view=Buttons())
        # clearall_button.callback = clearall_callback

        # clear10_button = discord.ui.Button(label="10", style=discord.ButtonStyle.primary)
        # view.add_item(clear10_button)
        # async def clear10_callback(interaction: discord.Interaction):
        #     await interaction.response.edit_message(embed=discord.Embed(title="Commands",description="Please choose one of the commands below by clicking the buttons.",color=discord.Color.orange()),view=Buttons())
        # clear10_button.callback = clear10_callback

        # clear20_button = discord.ui.Button(label="20", style=discord.ButtonStyle.primary)
        # view.add_item(clear20_button)
        # async def clear20_callback(interaction: discord.Interaction):
        #     await interaction.response.edit_message(embed=discord.Embed(title="Commands",description="Please choose one of the commands below by clicking the buttons.",color=discord.Color.orange()),view=Buttons())
        # clear20_button.callback = clear20_callback

        # clear50_button = discord.ui.Button(label="50", style=discord.ButtonStyle.primary)
        # view.add_item(clear50_button)
        # async def clear50_callback(interaction: discord.Interaction):
        #     await interaction.response.edit_message(embed=discord.Embed(title="Commands",description="Please choose one of the commands below by clicking the buttons.",color=discord.Color.orange()),view=Buttons())
        # clear50_button.callback = clear50_callback

        # clear100_button = discord.ui.Button(label="100", style=discord.ButtonStyle.primary)
        # view.add_item(clear100_button)
        # async def clear100_callback(interaction: discord.Interaction):
        #     await interaction.response.edit_message(embed=discord.Embed(title="Commands",description="Please choose one of the commands below by clicking the buttons.",color=discord.Color.orange()),view=Buttons())
        # clear100_button.callback = clear100_callback

        # clear250_button = discord.ui.Button(label="250", style=discord.ButtonStyle.primary)
        # view.add_item(clear250_button)
        # async def clear250_callback(interaction: discord.Interaction):
        #     await interaction.response.edit_message(embed=discord.Embed(title="Commands",description="Please choose one of the commands below by clicking the buttons.",color=discord.Color.orange()),view=Buttons())
        # clear250_button.callback = clear250_callback

        # clear500_button = discord.ui.Button(label="500", style=discord.ButtonStyle.primary)
        # view.add_item(clear500_button)
        # async def clear500_callback(interaction: discord.Interaction):
        #     await interaction.response.edit_message(embed=discord.Embed(title="Commands",description="Please choose one of the commands below by clicking the buttons.",color=discord.Color.orange()),view=Buttons())
        # clear500_button.callback = clear500_callback
        
        # back_button = discord.ui.Button(label="Back", style=discord.ButtonStyle.secondary)
        # view.add_item(back_button)
        # async def back_callback(interaction: discord.Interaction):
        #     await interaction.response.edit_message(embed=discord.Embed(title="Commands",description="Please choose one of the commands below by clicking the buttons.",color=discord.Color.orange()),view=Buttons())
        # back_button.callback = back_callback

        
        
        # await interaction.response.edit_message(embed=discord.Embed(title="Clear", description="Choose how many messages you want to erase", color=discord.Color.orange()), view=view)

    @discord.ui.button(label="Help", style=discord.ButtonStyle.secondary)
    async def help_callback(self, interaction: discord.Interaction, button: discord.ui.Button):
        embed = discord.Embed(title="Help", color=discord.Color.orange())
        embed.add_field(name="Announce", value="Announces a message to a specified channel.")
        embed.add_field(name="Clear", value="Clears messages from the channel.")
        
        back_button = discord.ui.Button(label="Back", style=discord.ButtonStyle.secondary)
        view = discord.ui.View()
        view.add_item(back_button)

        await interaction.response.edit_message(embed=embed, view=view)

        async def back_callback(interaction: discord.Interaction):
            await interaction.response.edit_message(embed=discord.Embed(title="Commands", description="Please choose one of the commands below by clicking the buttons.", color=discord.Color.orange()), view=Buttons())

        back_button.callback = back_callback

# async def is_channel(ctx):
#     return ctx.channel.id == 

# @bot.command()
# @commands.check(is_channel)
# async def announce(ctx, channel: discord.TextChannel, *, arg):
#     #channel = bot.get_channel() #announcements
#     await channel.send(arg)
#     await ctx.message.delete()

# @bot.command()
# @commands.check(is_channel)
# async def clear(ctx, amount: int = 0):
#     if amount:
#         await ctx.channel.purge(limit=amount + 1)
#     else:
#         await ctx.channel.purge()

# @announce.error
# async def announce_error(ctx, error):
#     if isinstance(error, commands.CheckFailure):
#         print('Invalid channel!')

@bot.command()
async def createinvite(ctx):
    invite = await bot.get_channel().create_invite(max_age=0, max_uses=0, unique=False)
    await ctx.send(f"{invite.url}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.channel.id == :
        await message.delete()

    await bot.process_commands(message)

@bot.event    
async def on_ready():
    print(f'Logged on as {bot.user}')
    
    channel = bot.get_channel()
    if channel:
        await channel.purge()
        message = await channel.send(embed=discord.Embed(description="Hello World!"))
        embed = discord.Embed(title="Commands", description="Please choose one of the commands below by clicking the buttons.",color=discord.Color.orange())
        await message.edit(embed=embed, view=Buttons())

bot.run('')