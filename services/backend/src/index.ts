import express from 'express';
import cors from 'cors';
import authRoutes from './routes/auth';
import companyRoutes from './routes/company';

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

app.use('/api/auth', authRoutes);
app.use('/api/company', companyRoutes);

app.get('/health', (req, res) => {
  res.send('OK');
});

app.listen(PORT, () => {
  console.log(`Backend server running on port ${PORT}`);
});
