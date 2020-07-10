package patztabot22

import java.io.File

class Pic(val path: String) {

  val list = { text: String ->
    println("LISTING $path")
    val fw = File(path).walk()
    fw.maxDepth(1)
    val it = fw.iterator()
    while (it.hasNext()) {
      val fd = it.next()
      println(fd.nameWithoutExtension)
    }
    0
  }

}
