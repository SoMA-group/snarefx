{if ($0 ~ /index|default|minimum|maximum|Version/) {gsub(/"/, "", $0); print $0} else {print $0}}
