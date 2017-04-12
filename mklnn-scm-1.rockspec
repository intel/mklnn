package = "mklnn"
version = "scm-1"

source = {
   url = "git://github.com/xhzhao/mklnn.git",
}

description = {
   summary = "Neural Network package for Torch",
   detailed = [[
   ]],
   homepage = "https://github.com/xhzhao/mklnn",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "luaffi"
}

build = {
   type = "command",
   build_command = [[
   cd build && cmake CMakeLists.txt && make
]],
   install_command = "cd build && $(MAKE) install"
}
