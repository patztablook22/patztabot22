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
      if (fd.extension.equals("")) {
        continue;
      }
      println(" - ${fd.nameWithoutExtension}")
    }
    0
  }

  fun find(name: String) : File?
  {
    val fw = File(path).walk()
    fw.maxDepth(1)
    val it = fw.iterator()
    while (it.hasNext()) {
      val fd = it.next()
      if (fd.extension.equals("")) {
        continue
      }
      if (fd.nameWithoutExtension.equals(name)) {
        return fd
      }
    }
    return null
  }

}
