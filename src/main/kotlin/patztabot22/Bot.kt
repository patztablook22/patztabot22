package patztabot22

import net.dv8tion.jda.api.JDABuilder
import net.dv8tion.jda.api.JDA
import net.dv8tion.jda.api.entities.*
import net.dv8tion.jda.api.hooks.*
import net.dv8tion.jda.api.events.message.MessageReceivedEvent

class Bot(val jda: JDA, val picc: Pic) : ListenerAdapter() {

  var index: List<TextChannel> = genIndex()
  var now: Int = -1

  override fun onMessageReceived( e: MessageReceivedEvent )
  {
    var msg = e.getMessage()
    var guild = e.getGuild()
    var chat = e.getChannel()
    var nick = e.getMember()!!.getEffectiveName()
    var text = msg.getContentRaw()
    val i = index.indexOfFirst { it == chat } + 1
    val j = i.toString()
    val len = j.length + nick.length + guild.getName().length + chat.getName().length

    println(7.toChar())
    println("──($j)─{ $nick @ ${guild.getName()} -> ${chat.getName()} }" + "─".repeat(64 - len))
    println(" $text\n")
  }

  fun genIndex(): List<TextChannel>
  {
    val buf: MutableList<TextChannel> = mutableListOf<TextChannel>()
    val guilds = jda.getGuilds()
    for (g in guilds) {
      buf.addAll(g.getTextChannels())
    }
    return buf
  }

  fun select(i: Int)
  {
    now = i
  }

  val test = { text: String ->
    0
  }

  val lg = { text: String ->
    val guilds = jda.getGuilds()
    for (g in guilds) {
      println(" | ${g.getName()}")
    }
    0
  }

  val cc = { text: String ->
    try {
      var i = text.toInt()
      if (i < 0 || i > index.size) {
        throw Exception()
      }
      i--
      now = i
      0
    } catch (e: Exception) {
      1
    }
  }

  val pc = { text: String ->
    if (now == -1) {
      println("no channel selected")
    } else {
      println("(${now}) ${index[now].getGuild().getName()} -> ${index[now].getName()}")
    }
    0
  }

  val lc = { text: String ->
    genIndex()

    println(" #    0 *nil*")
    index.forEachIndexed { i, c ->
      print(" # ")
      val j = (i + 1).toString()
      print("${ " ".repeat(4 - j.length) }$j ")
      println("${c.getGuild().getName()} -> ${c.getName()}")
    }
    
    0
  }

  val msg = { text: String ->
    if (now == -1) {
      1
    } else {
      index[now].sendMessage(text).queue()
      0
    }
  }

  val pic = { text: String ->
    if (now == -1) {
      1
    } else {
      0
    }
  }

}
