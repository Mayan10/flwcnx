import { Router, Request, Response } from 'express';
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import { v4 as uuidv4 } from 'uuid';
import prisma from '../db';

const router = Router();
const JWT_SECRET = process.env.JWT_SECRET || 'supersecretjwtkeythatshouldbechangedinprod';

// Register
router.post('/register', async (req: Request, res: Response) => {
  const { name, domain, city, password } = req.body;

  if (!name || !domain || !city || !password) {
    return res.status(400).json({ error: 'All fields are required' });
  }

  try {
    const existingCompany = await prisma.company.findUnique({ where: { domain } });
    if (existingCompany) {
      return res.status(400).json({ error: 'Company with this domain already exists' });
    }

    const passwordHash = await bcrypt.hash(password, 10);
    const apiKey = `fcx_${uuidv4().replace(/-/g, '')}`;

    const newCompany = await prisma.company.create({
      data: {
        name,
        domain,
        city,
        passwordHash,
        apiKey,
      },
    });

    const token = jwt.sign({ companyId: newCompany.id }, JWT_SECRET, { expiresIn: '1d' });

    await prisma.activity.create({
      data: {
        companyId: newCompany.id,
        description: 'Company registered',
      }
    });

    res.status(201).json({ token, company: { id: newCompany.id, name: newCompany.name, domain: newCompany.domain, apiKey: newCompany.apiKey } });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Server error during registration' });
  }
});

// Login
router.post('/login', async (req: Request, res: Response) => {
  const { domain, password } = req.body;

  if (!domain || !password) {
    return res.status(400).json({ error: 'Domain and password are required' });
  }

  try {
    const company = await prisma.company.findUnique({ where: { domain } });
    if (!company) {
      return res.status(400).json({ error: 'Invalid credentials' });
    }

    const isMatch = await bcrypt.compare(password, company.passwordHash);
    if (!isMatch) {
      return res.status(400).json({ error: 'Invalid credentials' });
    }

    const token = jwt.sign({ companyId: company.id }, JWT_SECRET, { expiresIn: '1d' });
    
    await prisma.activity.create({
      data: {
        companyId: company.id,
        description: 'Company logged in via Web UI',
      }
    });

    res.json({ token, company: { id: company.id, name: company.name, domain: company.domain } });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Server error during login' });
  }
});

// Verify API Key (Used by TUI)
router.post('/verify-key', async (req: Request, res: Response) => {
  const { apiKey } = req.body;

  if (!apiKey) {
    return res.status(400).json({ error: 'API key is required' });
  }

  try {
    const company = await prisma.company.findUnique({ where: { apiKey } });
    
    if (!company) {
      return res.status(401).json({ error: 'Invalid API key' });
    }

    await prisma.activity.create({
      data: {
        companyId: company.id,
        description: 'TUI logged in via API key',
      }
    });

    res.json({ success: true, company: { name: company.name, domain: company.domain } });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Server error verifying API key' });
  }
});

export default router;
