package authz

default allow := false

allow if {
    input.method == "GET"
    input.path[0] == "public"
}

allow if {
    some role in input.user.roles
    data.admins[role]
}