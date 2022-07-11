# Plugins or Something

Some LV2 plugins for researching things.

* SexyZebra.lv2
  - SexyZebra10: A 10 band dynamic graphic EQ
  - SexyZebra30: A 30 band dynamic graphic EQ

## Dependencies 
To build everything you need the following stuff installed:


| Dependency  | APT Package     |
|-------------|-----------------|
| LV2 headers | lv2-dev         |
| libserd     | libserd-dev     |
| libsndfile  | libsndfile1-dev |

## Installation
Build and install everything using the usual cmake gumpf:

```shell
$ mkdir build && cd build
$ cmake ..
$ make
$ sudo make install
```

Plug-in bundles will be installed in /usr/local/lib/lv2 by default. If you want them to go somewhere else set the LV2_PATH
environment variable.

Once things are installed you can test they are working with `lv2ls`, `lv2info`, and friends (in the lilv-utils APT
package).  `lv2ls` should give you output something like this:

```
https://gitlab.com/dmt-soma/lv2_plugs#SexyZebra10
https://gitlab.com/dmt-soma/lv2_plugs#SexyZebra30
```
