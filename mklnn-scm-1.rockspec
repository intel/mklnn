package = "mklnn"
version = "scm-1"

source = {
   url = "git://github.com/intel/mklnn.git",
}

description = {
   summary = "Neural Network package for Torch",
   detailed = [[
   ]],
   homepage = "https://github.com/intel/mklnn",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "luaffi"
}

build = {
   type = "command",

   build_variables = {
   CFLAGS = "$(CFLAGS)",
   LIBFLAG = "$(LIBFLAG)",
   LUA_LIBDIR = "$(LUA_LIBDIR)",
   LUA_BINDIR = "$(LUA_BINDIR)",
   LUA = "$(LUA)",
},

   install_variables = {
   INST_PREFIX = "$(PREFIX)",
   INST_BINDIR = "$(BINDIR)",
   INST_LIBDIR = "$(LIBDIR)",
   INST_LUADIR = "$(LUADIR)",
   INST_CONFDIR = "$(CONFDIR)",
},

   build_command = [[
   cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}
