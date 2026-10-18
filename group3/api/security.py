import os
from datetime import datetime, timedelta, timezone
from uuid import UUID
import jwt
from fastapi import Depends, Header, HTTPException
from .contracts import Role, User

SECRET=os.getenv("JWT_SECRET","replace-in-development")
TOKEN_MINUTES=int(os.getenv("JWT_EXPIRES_MINUTES","60"))
def issue(user: User):
    now=datetime.now(timezone.utc)
    return jwt.encode({"sub":str(user.id),"email":user.email,"name":user.name,"role":user.role.value,"iat":now,"exp":now+timedelta(minutes=TOKEN_MINUTES)},SECRET,algorithm="HS256")
def decode_token(token: str) -> User:
    try:
        p=jwt.decode(token,SECRET,algorithms=["HS256"]); return User(id=UUID(p["sub"]),email=p["email"],name=p["name"],role=Role(p["role"]))
    except (jwt.InvalidTokenError,KeyError,ValueError) as e: raise HTTPException(401,detail={"code":"INVALID_TOKEN","message":"Invalid token"}) from e
def current_user(authorization: str|None=Header(default=None)) -> User:
    if not authorization or not authorization.startswith("Bearer "): raise HTTPException(401,detail={"code":"AUTH_REQUIRED","message":"Bearer token required"})
    return decode_token(authorization[7:])
def allow(*roles: Role):
    def check(user: User=Depends(current_user)):
        if user.role not in roles: raise HTTPException(403,detail={"code":"FORBIDDEN","message":"Role is not allowed for this operation"})
        return user
    return check
