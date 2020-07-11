package patztabot22

import net.dv8tion.jda.api.JDABuilder
import net.dv8tion.jda.api.JDA
import net.dv8tion.jda.api.entities.*

fun main( args: Array<String> )
{

  val handler = Thread() {
      println("\nshutting down")
  }

  val run = Runtime.getRuntime()
  run.addShutdownHook(handler)

  if (args.size != 1) {
    println("wrong number of arguments")
    return
  }

  val jda: JDA

  try {

    println("connecting...")
    val builder = JDABuilder.createDefault(args[0])
      .setActivity(Activity.playing("ludus mundi"))
    jda = builder.build()

  } catch (e: javax.security.auth.login.LoginException) {
    println("login failed")
    return
  } catch (e: java.net.UnknownHostException) {
    println("network error")
    // jda doesnt throw the exception actually qq
    return
  }

  jda.awaitReady()
  val pic = Pic("/home/patz/Keep/pic/")
  val bot = Bot(jda, pic)
  jda.addEventListener(bot)

  prompt@ while (true) {

    print("\nINPUT> ")
    var input = readLine()!!

    val arr = input.split(" ")
    if (arr.size == 0 || arr[0].length == 0) {
      continue
    }
    val space: Int
    if (arr.size > 1) {
      space = 1
    } else {
      space = 0
    }
    input = input.substring( arr[0].length + space, input.length )

    val todo = when (arr[0]) {
      "test"  -> bot.test
      "cc"    -> bot.cc
      "pc"    -> bot.pc
      "lc"    -> bot.lc
      "lg"    -> bot.lg
      "m", "msg"
              -> bot.msg
      "pic"   -> bot.pic
      "pics"  -> pic.list
      "link"  -> bot.link
      "quit", "exit"
              -> break@prompt
      "leave" -> bot.leave
      else    -> { text: String -> println("nil"); 1 }
    }

    val rtn: Int = todo(input)

    when (rtn) {
      0 -> println("OK~~")
      else -> println("ERR~~")
    }

  }

  jda.shutdown()

}
