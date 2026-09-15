import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET || 'supersecretjwtkeythatshouldbechangedinprod';

export interface AuthRequest extends Request {
  companyId?: string;
}

export const authenticate = (req: AuthRequest, res: Response, next: NextFunction) => {
  const token = req.header('Authorization')?.replace('Bearer ', '');

  if (!token) {
    return res.status(401).json({ error: 'Access denied. No token provided.' });
  }

  try {
    const decoded = jwt.verify(token, JWT_SECRET) as { companyId: string };
    req.companyId = decoded.companyId;
    next();
  } catch (ex) {
    res.status(400).json({ error: 'Invalid token.' });
  }
};
